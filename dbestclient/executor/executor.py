# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

import os
import os.path
import warnings
from datetime import datetime
from multiprocessing import set_start_method as set_start_method_cpu

import dill
import numpy as np
import torch
from dbestclient.catalog.catalog import DBEstModelCatalog
from dbestclient.executor.queryengine import QueryEngine
from dbestclient.executor.queryenginemdn import (
    MdnQueryEngine,
    MdnQueryEngineGoGs,
    MdnQueryEngineNoRange,
    MdnQueryEngineNoRangeCategorical,
    MdnQueryEngineNoRangeCategoricalOneModel,
    MdnQueryEngineXCategorical,
    MdnQueryEngineXCategoricalOneModel,
    MdnQueryEngineRangeNoCategorical,
)
from dbestclient.io.sampling import DBEstSampling
from dbestclient.ml.modeltrainer import (
    GroupByModelTrainer,
    KdeModelTrainer,
    SimpleModelTrainer,
)
from dbestclient.ml.modelwraper import GroupByModelWrapper, get_pickle_file_name
from dbestclient.parser.parser import DBEstParser
from dbestclient.tools.dftools import (
    get_group_count_from_df,
    get_group_count_from_summary_file,
    get_group_count_from_table,
)
from dbestclient.tools.running_parameters import RUNTIME_CONF, DbestConfig
from dbestclient.tools.variables import Slave, UseCols
from torch.multiprocessing import set_start_method as set_start_method_torch


class SqlExecutor:
    """
    This is the executor for the SQL query.
    """

    def __init__(self):
        self.parser = None
        self.config = DbestConfig()  # model-related configuration
        self.runtime_config = RUNTIME_CONF
        self.last_config = None
        self.model_catalog = DBEstModelCatalog()
        self.init_slaves()
        self.init_model_catalog()

        self.save_sample = False
        # self.table_header = None
        self.n_total_records = None
        self.use_kde = True

    def init_model_catalog(self):
        # search the warehouse, and add all available models.
        n_model = 0
        t1 = datetime.now()
        # print("current directory is ",os.getcwd())
        for file_name in os.listdir(self.config.get_config()["warehousedir"]):
            # load simple models
            if file_name.endswith(self.runtime_config["model_suffix"]):
                if n_model == 0:
                    print("start loading pre-existing models.")

                with open(
                    self.config.get_config()["warehousedir"] + "/" + file_name, "rb"
                ) as f:
                    model = dill.load(f)
                self.model_catalog.model_catalog[
                    model.init_pickle_file_name(self.runtime_config)
                ] = model
                n_model += 1

        if n_model > 0:
            print("Loaded " + str(n_model) + " models.", end=" ")
            if self.runtime_config["b_show_latency"]:
                t2 = datetime.now()
                print("time cost ", (t2 - t1).total_seconds(), "s")
            else:
                print()

    def init_slaves(self):
        file_name = os.path.join(self.config.config["warehousedir"], "slaves")
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, "r") as f:
                for line in f:
                    if "#" not in line:
                        self.runtime_config["slaves"].add(Slave(line))
            if self.runtime_config["v"]:
                print(
                    "Cluster mode is on, slaves are "
                    + self.runtime_config["slaves"].to_string()
                )
        else:
            if self.runtime_config["v"]:
                print("Local mode is on, as no slaves are provided.")

    def execute(self, sql):
        # prepare the parser
        if type(sql) == str:
            self.parser = DBEstParser()
            self.parser.parse(sql)
        elif type(sql) == DBEstParser:
            self.parser = sql
        else:
            print("Unrecognized SQL! Please check it!")
            exit(-1)

        # execute the query
        if self.parser.if_nested_query():
            warnings.warn("Nested query is currently not supported!")
        else:
            sql_type = self.parser.get_query_type()
            if sql_type == "create":  # process create query
                # initialize the configure for each model creation.
                if self.last_config:
                    self.config = self.last_config
                else:
                    self.config = DbestConfig()
                # DDL, create the model as requested
                mdl = self.parser.get_ddl_model_name()
                tbl = self.parser.get_from_name()

                # remove unnecessary charactor '
                tbl = tbl.replace("'", "")
                if os.path.isfile(tbl):  # the absolute path is provided
                    original_data_file = tbl
                else:  # the file is in the warehouse direcotry
                    original_data_file = (
                        self.config.get_config()["warehousedir"] + "/" + tbl
                    )
                yheader = self.parser.get_y()

                xheader_continous, xheader_categorical = self.parser.get_x()

                ratio = self.parser.get_sampling_ratio()
                method = self.parser.get_sampling_method()
                table_header = self.config.get_config()["table_header"]
                # print("table_header", table_header)
                if table_header is not None:
                    table_header = table_header.split(
                        self.config.get_config()["csv_split_char"]
                    )
                if xheader_continous:  # there is no continuous attribute
                    if self.parser.if_model_need_filter():
                        self.config.set_parameter("accept_filter", True)

                # make samples
                if not self.parser.if_contain_groupby():  # if group by is not involved
                    sampler = DBEstSampling(
                        headers=table_header,
                        usecols={
                            "y": yheader,
                            "x_continous": xheader_continous,
                            "x_categorical": xheader_categorical,
                            "gb": None,
                        },
                    )
                else:
                    groupby_attribute = self.parser.get_groupby_value()

                    sampler = DBEstSampling(
                        headers=table_header,
                        usecols={
                            "y": yheader,
                            "x_continous": xheader_continous,
                            "x_categorical": xheader_categorical,
                            "gb": groupby_attribute,
                        },
                    )

                # print(self.config)
                if os.path.exists(
                    os.path.join(
                        self.config.get_config()["warehousedir"],
                        mdl + self.runtime_config["model_suffix"],
                    )
                ):
                    print(
                        "Model {0} exists in the warehouse, please use"
                        " another model name to train it.".format(mdl)
                    )
                    return
                # if self.parser.if_contain_groupby():
                #     groupby_attribute = self.parser.get_groupby_value()
                #     if os.path.exists(self.config['warehousedir'] + "/" + mdl + "_groupby_" + groupby_attribute):
                #         print(
                #             "Model {0} exists in the warehouse, please use"
                #             " another model name to train it.".format(mdl))
                #         return
                print("Start creating model " + mdl)
                time1 = datetime.now()

                # if method.lower() == "uniform":
                if self.save_sample:
                    sampler.make_sample(
                        original_data_file,
                        ratio,
                        method,
                        split_char=self.config.get_config()["csv_split_char"],
                        file2save=self.config.get_config()["warehousedir"]
                        + "/"
                        + mdl
                        + ".csv",
                        num_total_records=self.n_total_records,
                    )
                else:
                    sampler.make_sample(
                        original_data_file,
                        ratio,
                        method,
                        split_char=self.config.get_config()["csv_split_char"],
                        num_total_records=self.n_total_records,
                    )

                # set the n_total_point and scaling factor for each model.
                # self.config.set_parameter(
                #     "n_total_point", sampler.n_total_point)
                # self.config.set_parameter(
                #     "scaling_factor", sampler.scaling_factor)
                # print("scaling_factor is ", sampler.scaling_factor)

                if not self.parser.if_contain_groupby():  # if group by is not involved

                    # n_total_point = sampler.n_total_point
                    # xys = sampler.getyx(yheader, xheader_continous)

                    # simple_model_wrapper = SimpleModelTrainer(mdl, tbl, xheader_continous, yheader,
                    #                                           n_total_point, ratio, config=self.config.copy()).fit_from_df(
                    #     xys, self.runtime_config)

                    # reg = simple_model_wrapper.reg
                    # density = simple_model_wrapper.density
                    # n_sample_point = int(simple_model_wrapper.n_sample_point)
                    # n_total_point = int(simple_model_wrapper.n_total_point)
                    # x_min_value = float(simple_model_wrapper.x_min_value)
                    # x_max_value = float(simple_model_wrapper.x_max_value)
                    # query_engine = QueryEngine(mdl, reg, density, n_sample_point,
                    #                            n_total_point, x_min_value, x_max_value, xheader_continous[
                    #                                0],
                    #                            self.config)
                    sampler.sample.sampledf["dummy_gb"] = "dummy"
                    sampler.sample.usecols = {
                        "y": yheader,
                        "x_continous": xheader_continous,
                        "x_categorical": xheader_categorical,
                        "gb": "dummy_gb",
                    }
                    n_total_point, xys = sampler.get_groupby_frequency_data()
                    # if not n_total_point['if_contain_x_categorical']:
                    n_total_point.pop("if_contain_x_categorical")
                    kdeModelWrapper = KdeModelTrainer(
                        mdl,
                        tbl,
                        xheader_continous[0],
                        yheader,
                        groupby_attribute=["dummy_gb"],
                        groupby_values=list(n_total_point.keys()),
                        n_total_point=n_total_point,
                        x_min_value=-np.inf,
                        x_max_value=np.inf,
                        config=self.config.copy(),
                    ).fit_from_df(
                        xys["data"], self.runtime_config, network_size="large"
                    )

                    qe_mdn = MdnQueryEngine(kdeModelWrapper, config=self.config.copy())

                    qe_mdn.serialize2warehouse(
                        self.config.get_config()["warehousedir"], self.runtime_config
                    )
                    self.model_catalog.add_model_wrapper(qe_mdn, self.runtime_config)

                else:  # if group by is involved in the query
                    if self.config.get_config()["reg_type"] == "qreg":
                        xys = sampler.getyx(yheader, xheader_continous)
                        n_total_point = get_group_count_from_table(
                            original_data_file,
                            groupby_attribute,
                            sep=self.config.get_config()["csv_split_char"],
                            headers=table_header,
                        )

                        n_sample_point = get_group_count_from_df(xys, groupby_attribute)
                        groupby_model_wrapper = GroupByModelTrainer(
                            mdl,
                            tbl,
                            xheader_continous,
                            yheader,
                            groupby_attribute,
                            n_total_point,
                            n_sample_point,
                            x_min_value=-np.inf,
                            x_max_value=np.inf,
                            config=self.config.copy(),
                        ).fit_from_df(xys, self.runtime_config)
                        groupby_model_wrapper.serialize2warehouse(
                            self.config.get_config()["warehousedir"]
                            + "/"
                            + groupby_model_wrapper.dir
                        )
                        self.model_catalog.model_catalog[
                            groupby_model_wrapper.dir
                        ] = groupby_model_wrapper.models
                    else:  # "mdn"
                        if method.lower() == "uniform":
                            xys = sampler.getyx(
                                yheader, xheader_continous, groupby=groupby_attribute
                            )

                            # xys[groupby_attribute] = pd.to_numeric(xys[groupby_attribute], errors='coerce')
                            # xys=xys.dropna(subset=[yheader, xheader,groupby_attribute])

                            # n_total_point = get_group_count_from_table(
                            #     original_data_file, groupby_attribute, sep=',',#self.config['csv_split_char'],
                            #     headers=self.table_header)
                            if isinstance(ratio, str):
                                frequency_file = (
                                    self.config.get_config()["warehousedir"]
                                    + "/"
                                    + ratio
                                )
                                # "/num_of_points.csv"
                                if os.path.exists(frequency_file):
                                    n_total_point = get_group_count_from_summary_file(
                                        frequency_file, sep=","
                                    )
                                    (
                                        n_total_point_sample,
                                        xys,
                                    ) = sampler.get_groupby_frequency_data()
                                    n_total_point[
                                        "if_contain_x_categorical"
                                    ] = n_total_point_sample["if_contain_x_categorical"]
                                else:
                                    raise FileNotFoundError(
                                        "scaling factor should come from the "
                                        + ratio
                                        + " in the warehouse folder, as"
                                        " stated in the SQL. However, the file is not found."
                                    )
                            else:
                                (
                                    n_total_point,
                                    xys,
                                ) = sampler.get_groupby_frequency_data()
                                # print(n_total_point)
                                # for cases when the data file is treated as a sample, we need to scale up the frequency for each group.
                                if ratio > 1:
                                    file_size = sampler.n_total_point
                                    ratio = float(ratio) / file_size
                                # if 0 < ratio < 1:
                                scaled_n_total_point = {}
                                if "if_contain_x_categorical" in n_total_point:
                                    scaled_n_total_point[
                                        "if_contain_x_categorical"
                                    ] = n_total_point.pop("if_contain_x_categorical")
                                if "categorical_distinct_values" in n_total_point:
                                    scaled_n_total_point[
                                        "categorical_distinct_values"
                                    ] = n_total_point.pop("categorical_distinct_values")
                                if "x_categorical_columns" in n_total_point:
                                    scaled_n_total_point[
                                        "x_categorical_columns"
                                    ] = n_total_point.pop("x_categorical_columns")
                                for key in n_total_point:
                                    # print("key", key, n_total_point[key])

                                    if not isinstance(n_total_point[key], dict):
                                        scaled_n_total_point[key] = (
                                            n_total_point[key] / ratio
                                        )
                                    else:
                                        scaled_n_total_point[key] = {}
                                        for sub_key in n_total_point[key]:
                                            scaled_n_total_point[key][sub_key] = (
                                                n_total_point[key][sub_key] / ratio
                                            )
                                n_total_point = scaled_n_total_point
                                # print("scaled_n_total_point", scaled_n_total_point)
                        elif method.lower() == "stratified":
                            pass
                        else:
                            raise TypeError("unexpected method")
                        # no continuous x attributes, which means there is not range predicate on continuous attribute
                        if not xheader_continous:
                            # use one model to support all categorical attribute
                            if self.config.config["one_model"]:
                                qe = MdnQueryEngineNoRangeCategoricalOneModel(
                                    self.config.copy()
                                )
                                usecols = {
                                    "y": yheader,
                                    "x_continous": xheader_continous,
                                    "x_categorical": xheader_categorical,
                                    "gb": groupby_attribute,
                                }
                                if method.lower() == "uniform":
                                    useCols = UseCols(usecols)

                                    # get the training data from samples.
                                    (
                                        gbs,
                                        xs,
                                        ys,
                                    ) = useCols.get_gb_x_y_cols_for_one_model()
                                    (
                                        gbs_data,
                                        xs_data,
                                        ys_data,
                                    ) = sampler.sample.get_columns_from_original_sample(
                                        gbs, xs, ys
                                    )
                                    n_total_point = sampler.sample.get_frequency_of_categorical_columns_for_gbs(
                                        groupby_attribute, xheader_categorical
                                    )
                                    # print("n_total_point-----------before",
                                    #       n_total_point)
                                    # print("ratio is ", ratio)

                                    scaled_n_total_point = {}
                                    for key in n_total_point:
                                        scaled_n_total_point[key] = {}
                                        for sub_key in n_total_point[key]:
                                            scaled_n_total_point[key][sub_key] = (
                                                n_total_point[key][sub_key] / ratio
                                            )
                                    n_total_point = scaled_n_total_point
                                    # print("n_total_point-----------after",
                                    #       n_total_point)

                                elif method.lower() == "stratified":
                                    (
                                        gbs_data,
                                        xs_data,
                                        ys_data,
                                    ) = sampler.sample.get_categorical_features_label()
                                    n_total_point = sampler.sample.get_ft()

                                # print("gbs_data", gbs_data)
                                # print("xs_data", xs_data)
                                # print("ys_data", ys_data)
                                # print("n_total_point", n_total_point)
                                # print("n_total_point['']", n_total_point[''])

                                # print("type", type(gbs_data))
                                # exit()

                                qe.fit(
                                    mdl,
                                    tbl,
                                    gbs_data,
                                    xs_data,
                                    ys_data,
                                    n_total_point,
                                    usecols=usecols,
                                    runtime_config=self.runtime_config,
                                )
                                qe.serialize2warehouse(
                                    self.config.get_config()["warehousedir"],
                                    self.runtime_config,
                                )
                                self.model_catalog.add_model_wrapper(
                                    qe, self.runtime_config
                                )
                            else:  # train seperate models for each categorical attribute
                                # print(xheader_continous,
                                #       xheader_categorical, "---------->")

                                usecols = {
                                    "y": yheader,
                                    "x_continous": xheader_continous,
                                    "x_categorical": xheader_categorical,
                                    "gb": groupby_attribute,
                                }

                                if (
                                    not xheader_categorical
                                ):  # For WHERE clause without categorical equality
                                    n_total_point.pop("if_contain_x_categorical")
                                    qe_mdn = MdnQueryEngineNoRange(
                                        config=self.config.copy()
                                    )
                                    qe_mdn.fit(
                                        mdl,
                                        tbl,
                                        xys["data"],
                                        n_total_point,
                                        usecols,
                                        self.runtime_config,
                                    )
                                else:  # For WHERE clause with categorical equality
                                    # print(xys)
                                    qe_mdn = MdnQueryEngineNoRangeCategorical(
                                        config=self.config.copy()
                                    )
                                    qe_mdn.fit(
                                        mdl,
                                        tbl,
                                        xys,
                                        n_total_point,
                                        usecols,
                                        self.runtime_config,
                                    )
                                qe_mdn.serialize2warehouse(
                                    self.config.get_config()["warehousedir"],
                                    self.runtime_config,
                                )
                                # kdeModelWrapper.serialize2warehouse()
                                self.model_catalog.add_model_wrapper(
                                    qe_mdn, self.runtime_config
                                )
                        else:
                            if method.lower() == "uniform":
                                if not n_total_point["if_contain_x_categorical"]:
                                    if not self.config.get_config()["b_use_gg"]:
                                        n_total_point.pop("if_contain_x_categorical")
                                        # print(" xys['data']", xys["data"])
                                        # exit()
                                        # xys.pop("if_contain_x_categorical")
                                        kdeModelWrapper = KdeModelTrainer(
                                            mdl,
                                            tbl,
                                            xheader_continous[0],
                                            yheader,
                                            groupby_attribute=groupby_attribute,
                                            groupby_values=list(n_total_point.keys()),
                                            n_total_point=n_total_point,
                                            x_min_value=-np.inf,
                                            x_max_value=np.inf,
                                            config=self.config.copy(),
                                        ).fit_from_df(
                                            xys["data"],
                                            self.runtime_config,
                                            network_size=None,
                                        )

                                        qe_mdn = MdnQueryEngine(
                                            kdeModelWrapper, config=self.config.copy()
                                        )
                                        qe_mdn.serialize2warehouse(
                                            self.config.get_config()["warehousedir"],
                                            self.runtime_config,
                                        )
                                        # kdeModelWrapper.serialize2warehouse()
                                        self.model_catalog.add_model_wrapper(
                                            qe_mdn, self.runtime_config
                                        )

                                    else:
                                        # print("n_total_point ", n_total_point)
                                        queryEngineBundle = MdnQueryEngineGoGs(
                                            config=self.config.copy()
                                        ).fit(
                                            xys["data"],
                                            groupby_attribute,
                                            n_total_point,
                                            mdl,
                                            tbl,
                                            xheader_continous[0],
                                            yheader,
                                            self.runtime_config,
                                        )  # n_per_group=n_per_gg,n_mdn_layer_node = n_mdn_layer_node,encoding = encoding,b_grid_search = b_grid_search

                                        self.model_catalog.add_model_wrapper(
                                            queryEngineBundle, self.runtime_config
                                        )
                                        queryEngineBundle.serialize2warehouse(
                                            self.config.get_config()["warehousedir"],
                                            self.runtime_config,
                                        )
                                else:  # x has categorical attributes
                                    # if not self.config.get_config()["b_use_gg"]:
                                    # use a single model to support categorical conditions.
                                    if self.config.config["one_model"]:
                                        qe = MdnQueryEngineXCategoricalOneModel(
                                            self.config.copy()
                                        )
                                        usecols = {
                                            "y": yheader,
                                            "x_continous": xheader_continous,
                                            "x_categorical": xheader_categorical,
                                            "gb": groupby_attribute,
                                        }
                                        useCols = UseCols(usecols)

                                        # get the training data from samples.
                                        (
                                            gbs,
                                            xs,
                                            ys,
                                        ) = useCols.get_gb_x_y_cols_for_one_model()
                                        (
                                            gbs_data,
                                            xs_data,
                                            ys_data,
                                        ) = sampler.sample.get_columns_from_original_sample(
                                            gbs, xs, ys
                                        )
                                        n_total_point = sampler.sample.get_frequency_of_categorical_columns_for_gbs(
                                            groupby_attribute, xheader_categorical
                                        )
                                        # print("n_total_point-----------before",
                                        #       n_total_point)
                                        # print("ratio is ", ratio)

                                        # print("gbs_data", gbs_data)
                                        # print("xs_data", xs_data)
                                        # print("ys_data", ys_data)
                                        # print("n_total_point", n_total_point)
                                        # print(
                                        #     "n_total_point['']", n_total_point.keys())
                                        # exit()

                                        scaled_n_total_point = {}
                                        for key in n_total_point:
                                            scaled_n_total_point[key] = {}
                                            for sub_key in n_total_point[key]:
                                                scaled_n_total_point[key][sub_key] = (
                                                    n_total_point[key][sub_key] / ratio
                                                )
                                        n_total_point = scaled_n_total_point
                                        # print("n_total_point-----------after",
                                        #       n_total_point)

                                        # raise

                                        qe.fit(
                                            mdl,
                                            tbl,
                                            gbs_data,
                                            xs_data,
                                            ys_data,
                                            n_total_point,
                                            usecols=usecols,
                                            runtime_config=self.runtime_config,
                                        )
                                    else:
                                        # print(xys)
                                        qe = MdnQueryEngineXCategorical(
                                            self.config.copy()
                                        )
                                        qe.fit(
                                            mdl,
                                            tbl,
                                            xys,
                                            n_total_point,
                                            usecols={
                                                "y": yheader,
                                                "x_continous": xheader_continous,
                                                "x_categorical": xheader_categorical,
                                                "gb": groupby_attribute,
                                            },
                                            runtime_config=self.runtime_config,
                                        )  # device=device, encoding=encoding, b_grid_search=b_grid_search
                                        qe.serialize2warehouse(
                                            self.config.get_config()["warehousedir"],
                                            self.runtime_config,
                                        )
                                        self.model_catalog.add_model_wrapper(
                                            qe, self.runtime_config
                                        )
                                        # else:
                                        #     raise ValueError(
                                        #         "GoG support for categorical attributes is not supported.")
                                    qe.serialize2warehouse(
                                        self.config.get_config()["warehousedir"],
                                        self.runtime_config,
                                    )
                                    self.model_catalog.add_model_wrapper(
                                        qe, self.runtime_config
                                    )
                            elif method.lower() == "stratified":
                                if xheader_categorical:

                                    # print("contain equality")
                                    # print("xheader_categorical",
                                    #       xheader_categorical)
                                    (
                                        gbs_data,
                                        xs_data,
                                        ys_data,
                                    ) = sampler.sample.get_categorical_features_label()
                                    n_total_point = sampler.sample.get_ft()
                                    usecols = {
                                        "y": yheader,
                                        "x_continous": xheader_continous,
                                        "x_categorical": xheader_categorical,
                                        "gb": groupby_attribute,
                                    }
                                    xs_data = xs_data.reshape(1, -1)[0]
                                    # print("gbs_data", gbs_data)
                                    # print("xs_data", xs_data)
                                    # print("ys_data", ys_data)
                                    # print("n_total_point", n_total_point)
                                    # print(
                                    #     "n_total_point['']", n_total_point.keys())

                                    # print("type", type(gbs_data))
                                    # exit()
                                    qe = MdnQueryEngineXCategoricalOneModel(
                                        self.config.copy()
                                    )
                                    qe.fit(
                                        mdl,
                                        tbl,
                                        gbs_data,
                                        xs_data,
                                        ys_data,
                                        n_total_point,
                                        usecols=usecols,
                                        runtime_config=self.runtime_config,
                                    )
                                    qe.serialize2warehouse(
                                        self.config.get_config()["warehousedir"],
                                        self.runtime_config,
                                    )
                                    self.model_catalog.add_model_wrapper(
                                        qe, self.runtime_config
                                    )
                                else:  # contain range, but not equality
                                    # print("does not contain equality")
                                    # print("xheader_categorical",
                                    #       xheader_categorical)
                                    (
                                        gbs_data,
                                        xs_data,
                                        ys_data,
                                    ) = sampler.sample.get_categorical_features_label()
                                    n_total_point = sampler.sample.get_ft()
                                    usecols = {
                                        "y": yheader,
                                        "x_continous": xheader_continous,
                                        "x_categorical": xheader_categorical,
                                        "gb": groupby_attribute,
                                    }
                                    xs_data = xs_data.reshape(1, -1)[0]

                                    # print(" xys['data']", xys["data"])
                                    # exit()

                                    # kdeModelWrapper = KdeModelTrainer(
                                    #     mdl, tbl, xheader_continous[0], yheader,
                                    #     groupby_attribute=groupby_attribute,
                                    #     groupby_values=list(
                                    #         n_total_point.keys()),
                                    #     n_total_point=n_total_point,
                                    #     x_min_value=-np.inf, x_max_value=np.inf,
                                    #     config=self.config.copy()).fit_from_df(
                                    #     xys["data"], self.runtime_config, network_size=None)

                                    # qe_mdn = MdnQueryEngine(
                                    #     kdeModelWrapper, config=self.config.copy())

                                    # print("n_total_point", n_total_point)
                                    qe = MdnQueryEngineRangeNoCategorical(
                                        self.config.copy()
                                    )
                                    qe.fit(
                                        mdl,
                                        tbl,
                                        gbs_data,
                                        xs_data,
                                        ys_data,
                                        n_total_point,
                                        usecols=usecols,
                                        runtime_config=self.runtime_config,
                                    )
                                    qe.serialize2warehouse(
                                        self.config.get_config()["warehousedir"],
                                        self.runtime_config,
                                    )
                                    self.model_catalog.add_model_wrapper(
                                        qe, self.runtime_config
                                    )

                            else:
                                raise TypeError("unexpected sampling method.")
                time2 = datetime.now()
                t = (time2 - time1).seconds
                if self.runtime_config["b_show_latency"]:
                    print("time cost: " + str(t) + "s.")
                print("------------------------")

                # rest config
                self.last_config = None
                return

            elif sql_type == "select":  # process SELECT query
                start_time = datetime.now()
                predictions = None
                # DML, provide the prediction using models
                mdl = self.parser.get_from_name()
                gb_to_print, [
                    func,
                    yheader,
                    distinct_condition,
                ] = self.parser.get_dml_aggregate_function_and_variable()
                # for query with WHERE clause containing range selector
                if (
                    self.parser.if_where_exists()
                    and self.parser.get_dml_where_categorical_equal_and_range()[2]
                ):

                    print("OK")
                    where_conditions = (
                        self.parser.get_dml_where_categorical_equal_and_range()
                    )
                    # print("where_conditions", where_conditions)
                    # xheader, x_lb, x_ub = self.parser.get_dml_where_categorical_equal_and_range()
                    if (
                        mdl + self.runtime_config["model_suffix"]
                        not in self.model_catalog.model_catalog
                    ):
                        print("Model " + mdl + " does not exist.")
                        return
                    model = self.model_catalog.model_catalog[
                        mdl + self.runtime_config["model_suffix"]
                    ]
                    x_header_density = model.density_column

                    # print("where_conditions", where_conditions)
                    [x_lb, x_ub] = [
                        where_conditions[2][x_header_density][i] for i in [0, 1]
                    ]
                    filter_dbest = dict(where_conditions[2])
                    filter_dbest = [
                        filter_dbest[next(iter(filter_dbest))][i] for i in [0, 1]
                    ]
                    # print("where_conditions",where_conditions)
                    # print("filter_dbest",filter_dbest)

                    predictions = model.predicts(
                        func,
                        x_lb,
                        x_ub,
                        where_conditions,
                        self.runtime_config,
                        groups=None,
                        filter_dbest=filter_dbest,
                    )
                    # predictions = model.predict_one_pass(
                    #     func, x_lb, x_ub, n_jobs=n_jobs)
                elif func == "var":
                    print("var!!")
                    model = self.model_catalog.model_catalog[
                        mdl + self.runtime_config["model_suffix"]
                    ]
                    x_header_density = model.density_column
                    # print(x_header_density)
                    predictions = model.predicts(
                        "var", runtime_config=self.runtime_config
                    )
                    # return predictions
                else:  # for query without WHERE range selector clause
                    print("OK")
                    where_conditions = (
                        self.parser.get_dml_where_categorical_equal_and_range()
                    )
                    if (
                        mdl + self.runtime_config["model_suffix"]
                        not in self.model_catalog.model_catalog
                    ):
                        print("Model " + mdl + " does not exist.")
                        return
                    model = self.model_catalog.model_catalog[
                        mdl + self.runtime_config["model_suffix"]
                    ]
                    predictions = model.predicts(
                        func,
                        None,
                        None,
                        where_conditions,
                        self.runtime_config,
                        groups=None,
                        filter_dbest=None,
                    )

                if self.runtime_config["b_print_to_screen"]:
                    # print(predictions.to_csv(sep=',', index=False))  # sep='\t'
                    print(predictions.to_string(index=False))  # max_rows=5

                if self.runtime_config["b_show_latency"]:
                    end_time = datetime.now()
                    time_cost = (end_time - start_time).total_seconds()
                    print("Time cost: %.4fs." % time_cost)
                print("------------------------")
                return predictions

            elif sql_type == "set":  # process SET query
                if self.last_config:
                    self.config = self.last_config
                else:
                    self.config = DbestConfig()
                try:
                    key, value = self.parser.get_set_variable_value()
                    if key in self.config.get_config():
                        # check variable value before assignment
                        if key.lower() == "encoder":
                            value = value.lower()
                            if value not in ["onehot", "binary", "embedding"]:
                                value = "binary"
                                print(
                                    "encoder is not set to a proper value, use default encoding type: binary."
                                )

                        self.config.get_config()[key] = value
                        print("OK, " + key + " is updated.")
                    else:  # if variable is within runtime_config
                        # check if "device" is set. we need to make usre when GPU is not availabe, cpu is used instead.
                        if key.lower() == "device":
                            value = value.lower()
                            if value in ["cpu", "gpu"]:
                                if torch.cuda.is_available():
                                    if value == "gpu":
                                        value = "cuda:0"
                                        try:
                                            set_start_method_torch("spawn")
                                        except RuntimeError:
                                            print(
                                                "Fail to set start method as spawn for pytorch multiprocessing, "
                                                + "use default in advance. (see queryenginemdn "
                                                "for more info.)"
                                            )
                                    else:
                                        set_start_method_cpu("spawn")
                                    if self.runtime_config["v"]:
                                        print("device is set to " + value)
                                else:
                                    if value == "gpu":
                                        print("GPU is not available, use CPU instead")
                                        value = "cpu"
                                    if value == "cpu":
                                        if self.runtime_config["v"]:
                                            print("device is set to " + value)
                            else:
                                print("Only GPU or CPU is supported.")
                                return

                        self.runtime_config[key] = value
                        if key in self.runtime_config:
                            print("OK, " + key + " is updated.")
                        else:
                            print("OK, local variable " + key + " is defined.")
                except TypeError:
                    # self.parser.get_set_variable_value() does not return correctly
                    print("Parameter is not changed. Please check your SQL!")

                # save the config
                self.last_config = self.config
                return

            elif sql_type == "drop":  # process DROP query
                model_name = self.parser.drop_get_model()
                model_path = os.path.join(
                    self.config.get_config()["warehousedir"],
                    model_name + self.runtime_config["model_suffix"],
                )
                if os.path.isfile(model_path):
                    os.remove(model_path)
                    print("OK. model is dropped.")
                    return True
                else:
                    print("Model does not exist!")
                    return False
            elif sql_type == "show":
                print("OK")
                t_start = datetime.now()
                if self.runtime_config["b_print_to_screen"]:
                    for key in self.model_catalog.model_catalog:
                        print(key.replace(self.runtime_config["model_suffix"], ""))
                if self.runtime_config["v"]:
                    t_end = datetime.now()
                    time_cost = (t_end - t_start).total_seconds()
                    print("Time cost: %.4fs." % time_cost)
            else:
                print("Unsupported query type, please check your SQL.")
                return

    def set_table_counts(self, dic):
        self.n_total_records = dic
