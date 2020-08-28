# Created by Qingzhi Ma at 2019-07-23
# All right reserved
# Department of Computer Science
# the University of Warwick
# Q.Ma.2@warwick.ac.uk

import re
import warnings

import sqlparse
from sqlparse import sql
# from sqlparse.sql import Function, Identifier, IdentifierList
# from sqlparse.sql import Identifier
from sqlparse.tokens import DDL, DML, Keyword

from dbestclient.tools.date import unix_timestamp

# from dataclasses import replace
# from os.path import abspath


class DBEstParser:
    """parse a single SQL query, of the following form:
    - **DDL**
        >>> CREATE TABLE t_m(y real, x real)
        >>> FROM tbl
        >>> [GROUP BY z]
        >>> [SIZE 0.01]
        >>> [METHOD UNIFROM|HASH]
        >>> [ENCODING ONEHOT|BINARY]
    - **DML**
        >>> SELECT AF(y)
        >>> FROM t_m
        >>> [WHERE x BETWEEN a AND b]
        >>> [GROUP BY z]

    .. note::
        - model name should be ended with **_m** to indicate that it is a model, not a table.
        - AF, or aggregate function, could be COUNT, SUM, AVG, VARIANCE, PERCENTILE, etc.
    """

    def __init__(self):
        self.query = ""
        self.parsed = None

    def parse(self, query):
        """
        parse a single SQL query, of the following form:

        - **DDL**
            >>> CREATE TABLE t_m(y real, x_1 real, ... x_n categorical)
            >>> FROM tbl
            >>> [GROUP BY z]
            >>> [SIZE 0.01]
            >>> [METHOD UNIFROM|HASH]

        - **DML**
            >>> SELECT AF(y)
            >>> FROM t_m
            >>> [WHERE x BETWEEN a AND b]
            >>> [GROUP BY z]

        - **parameters**
        :param query: a SQL query
        """

        self.query = re.sub(' +', ' ', query).replace(" (", "(").lstrip()
        if "between" in self.query.lower():
            raise ValueError(
                "BETWEEN clause is not supported, please use 0<=x<=10 instead.")
        self.parsed = sqlparse.parse(self.query)[0]

    def if_nested_query(self):
        idx = 0
        if not self.parsed.is_group:
            return False
        for item in self.parsed.tokens:
            if item.ttype is DML and item.value.lower() == 'select':
                idx += 1
        if idx > 1:
            return True
        return False

    def get_dml_aggregate_function_and_variable(self):
        values = self.parsed.tokens[2].normalized
        if "," in values:
            splits = values.split(",")
            # print(splits)
            y_splits = splits[1].replace(
                "(", " ").replace(")", " ")  # .split(" ")
            # print(y_splits)
            if "distinct" in y_splits.lower():
                y_splits = y_splits.split()
                # print(y_splits)
                return splits[0], [y_splits[i] for i in [0, 2, 1]]
            else:
                y_splits = y_splits.split()
                y_splits.append(None)
                return splits[0], y_splits
        else:
            y_splits = values.replace(
                "(", " ").replace(")", " ")
            if "distinct" in y_splits.lower():
                y_splits = y_splits.split()
                # print(y_splits)
                return None, [y_splits[i] for i in [0, 2, 1]]
            else:
                y_splits = y_splits.split()
                y_splits.append(None)
                return None, y_splits
        # for item in self.parsed.tokens:
        #     print(self.parsed.tokens[2].normalized)
        #     if item.ttype is DML and item.value.lower() == 'select':
        #         print(self.parsed.token_index)
        #         idx = self.parsed.token_index(item, 0) + 2
        #         return self.parsed.tokens[idx].tokens[0].value, \
        #             self.parsed.tokens[idx].tokens[1].value.replace(
        #                 "(", "").replace(")", "")

    def if_where_exists(self):
        for item in self.parsed.tokens:
            if 'where' in item.value.lower():
                return True
        return False

    # def get_where_x_and_range(self):
    #     for item in self.parsed.tokens:
    #         if 'where' in item.value.lower():
    #             print(item)
    #             print(item.tokens)
    #             whereclause = item.value.lower().split()
    #             idx = whereclause.index("between")
    #             # print(idx)
    #             return whereclause[idx-1], whereclause[idx+1], whereclause[idx+3]
    #             # return whereclause[1], whereclause[3], whereclause[5]

    # def get_where_x_and_range(self):
    #     for item in self.parsed.tokens:
    #         if 'where' in item.value.lower():
    #             for it in item.tokens:
    #                 # print(it.value, it.ttype)
    #                 if isinstance(it, sql.Comparison):
    #                     splits = it.value.replace("=", "").split("<")

    #                     return [splits[0], splits[2]]
    def drop_get_model(self):
        if self.get_query_type() != "drop":
            raise TypeError("This is not a DROP query, please check it.")
        else:
            for item in self.parsed.tokens:
                if isinstance(item, sql.Identifier):
                    return item.normalized

    def get_dml_where_categorical_equal_and_range(self):
        """ get the equal and range selection for categorical attributes.

        For example,

        321<X1 < 1123 and x2 = 'HaHaHa' and x3='' and x4<5 produces

        ['x2', 'x3'],

        ["'HaHaHa'", "''"],

        {'X1': ['321', '1123', False, False], 'x4': [None, 5.0, False, False]}

        Raises:
            ValueError: unexpected condition in SQL

        Returns:
            tuple: list, list, dict
        """
        equal_xs = []
        equal_values = []
        conditions = {}
        for item in self.parsed.tokens:
            clause_lower = item.value.lower().replace("( ", "(").replace(" )", ")")
            clause = item.value.replace("( ", "(").replace(" )", ")")
            if 'where' in clause_lower:

                # for token in item:
                #     print(token.is_group, token.is_keyword,
                #           token.is_whitespace, token.normalized)
                splits = clause.replace("=", " = ").replace(
                    "AND", "and").replace("where", "").split("and")
                # splits_lower = clause_lower.replace("=", " = ").split("and")

                # print("splits", splits)
                for condition in splits:
                    if any(pattern in condition for pattern in ["=", ">", "<"]):
                        condition = condition.replace(" ", "")
                        # firstly check if there is a bi-directinal condition or not.
                        count_less = condition.count("<")
                        if count_less == 2:  # 1<x<2

                            condition_no_equal = condition.replace("=", "")
                            splits = condition_no_equal.split("<")
                            if "unix_timestamp" in splits[0]:
                                left = unix_timestamp(splits[0].replace(
                                    "unix_timestamp(", "").replace(")", "").replace("'", "").replace('"', ''))
                            else:
                                left = float(splits[0])

                            if "unix_timestamp" in splits[2]:
                                right = unix_timestamp(splits[2].replace(
                                    "unix_timestamp(", "").replace(")", "").replace("'", "").replace('"', ''))
                            else:
                                right = float(splits[2])
                            cond = [left, right]
                            key = splits[1]

                            splits = condition.split(splits[1])
                            if "=" in splits[0]:  # 0 <= x <...
                                cond.append(True)
                            else:
                                cond.append(False)

                            if "=" in splits[1]:  # ...< x <=2
                                cond.append(True)
                            else:
                                cond.append(False)
                            conditions[key] = cond

                        else:
                            if "<=" in condition:
                                splits = condition.split("<=")
                                conditions[splits[0]] = [
                                    None, splits[1], False, True]
                            elif ">=" in condition:
                                splits = condition.split(">=")
                                conditions[splits[0]] = [
                                    splits[1], None, True, False]
                            elif "<" in condition:
                                splits = condition.split("<")
                                conditions[splits[0]] = [
                                    None, splits[1], False, False]
                            elif ">" in condition:
                                splits = condition.split(">")
                                conditions[splits[0]] = [
                                    splits[1], None, False, False]
                            elif "=" in condition:
                                splits = condition.split("=")
                                equal_xs.append(splits[0])
                                equal_values.append(splits[1])
                                # print(equal_xs, equal_values)
                            else:
                                raise ValueError(
                                    "unexpected condition in SQL: ", condition)
                return [equal_xs, equal_values, conditions]

                # # # = condition
                # # if "=" in condition and not any(pattern in condition for pattern in [">", "<"]):
                # #     print("only =")
                # #     equal_xs.append()
                # #     equal_values.append()
                # # # >= <= condition
                # # elif "=" in condition:
                # #     print(">=")
                # #     print(condition.count("="))
                # # # no = condition, which is > or <
                # # else:
                # #     print("no =")

                # # indexes = [m.start() for m in re.finditer('=', clause)]
                # splits = clause.replace("=", " = ").split()
                # splits_lower = clause_lower.replace("=", " = ").split()
                # # print(clause)
                # # print(clause.count("="))
                # xs = []
                # values = []
                # while True:
                #     if "=" not in splits:
                #         break
                #     idx = splits.index("=")
                #     xs.append(splits_lower[idx-1])
                #     if splits[idx+1] != "''":
                #         values.append(splits[idx+1].replace("'", ""))
                #     else:
                #         values.append("")
                #     splits = splits[idx+3:]
                #     splits_lower = splits_lower[idx+3:]
                # #     print(splits)
                # # print(xs, values)
                # return xs, values

    def if_contain_groupby(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "group by":
                return True
        return False

    # def if_contain_scaling_factor(self):
    #     for item in self.parsed.tokens:
    #         if item.ttype is Keyword and item.value.lower() == "scale":
    #             return True
    #     return False

    # def get_scaling_method(self):
    #     if not self.if_contain_scaling_factor():
    #         return "data"
    #     else:
    #         for item in self.parsed.tokens:
    #             if item.ttype is Keyword and item.value.lower() == "scale":
    #                 idx = self.parsed.token_index(item, 0) + 2
    #                 if self.parsed.tokens[idx].value.lower() not in ["data", "file"]:
    #                     raise ValueError(
    #                         "Scaling method is not set properly, wrong argument provided.")
    #                 else:

    #                     method = self.parsed.tokens[idx].value.lower()
    #                     if method == "file":
    #                         file = self.parsed.tokens[idx+2].value.lower()
    #                         return method, file
    #                     else:
    #                         return method, None

    def get_groupby_value(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "group by":
                idx = self.parsed.token_index(item, 0) + 2
                groups = self.parsed.tokens[idx].value
                return groups.replace(" ", "").split(",")

    def if_ddl(self):
        for item in self.parsed.tokens:
            if item.ttype is DDL and item.value.lower() == "create":
                return True
        return False

    def get_ddl_model_name(self):
        for item in self.parsed.tokens:
            if item.ttype is None and "(" in item.value.lower():
                return item.tokens[0].value

    def get_y(self):
        item = self.parsed.tokens[4].value
        index_comma = item.index(",")
        item = item[:index_comma]
        y_list = item.lower().replace(
            "(", " ").replace(")", " ").replace(",", " ").split()
        # print("y_list", y_list)
        if y_list[2] not in ["real", "categorical"]:
            raise TypeError("Unsupported type for " +
                            y_list[1] + " -> " + y_list[2])
        # if item.ttype is None and "(" in item.value.lower():
        #     y_list = item.tokens[1].value.lower().replace(
        #         "(", "").replace(")", "").replace(",", " ").split()
        #     if y_list[1] not in ["real", "categorical"]:
        #         raise TypeError("Unsupported type for " +
        #                         y_list[0] + " -> " + y_list[1])
        if len(y_list) == 4:
            return [y_list[1], y_list[2], y_list[3]]
        else:
            return [y_list[1], y_list[2], None]

        # return item.tokens[1].tokens[1].value, item.tokens[1].tokens[3].value

    # def get_x(self):
    #     item = self.parsed.tokens[4].value
    #     index_comma = item.index(",")
    #     item = item[index_comma+1:]
    #     x_list = item.lower().replace(
    #         "(", "").replace(")", "").replace(",", " ").split()
    #     # print(x_list)
    #     continous = []
    #     categorical = []
    #     for idx in range(1, len(x_list), 2):
    #         if x_list[idx] == "real":
    #             continous.append(x_list[idx-1])
    #         if x_list[idx] == "categorical":
    #             categorical.append(x_list[idx-1])

    #     if len(continous) > 1:
    #         raise SyntaxError(
    #             "Only one continous independent variable is supported at "
    #             "this moment, please modify your SQL query accordingly.")
    #     # print("continous,", continous)
    #     # print("categorical,", categorical)
    #     return continous, categorical

    def get_x(self):
        item = self.parsed.tokens[4].value
        index_comma = item.index(",")
        item = item[index_comma+1:]
        x_list = item.lower().replace(
            "(", "").replace(")", "").replace(",", " ").split()
        # print(x_list)
        continous = []
        categorical = []
        for idx in range(1, len(x_list), 2):
            if x_list[idx] == "real":
                continous.append(x_list[idx-1])
            if x_list[idx] == "categorical":
                categorical.append(x_list[idx-1])

        if len(continous) > 1:
            raise SyntaxError(
                "Only one continous independent variable is supported at "
                "this moment, please modify your SQL query accordingly.")
        # print("continous,", continous)
        # print("categorical,", categorical)
        return continous, categorical

    def get_from_name(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "from":
                idx = self.parsed.token_index(item, 0) + 2
                return self.parsed.tokens[idx].value

    def get_sampling_ratio(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "size":
                idx = self.parsed.token_index(item, 0) + 2
                value = self.parsed.tokens[idx].value
                try:
                    value_float = float(value)
                    if "." not in value:
                        value = int(value_float)
                    else:
                        value = float(value)
                except ValueError:
                    value = value.replace("'", "")
                return value
        return 1  # if sampling ratio is not passed, the whole dataset will be used to train the model

    def get_sampling_method(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "method":
                idx = self.parsed.token_index(item, 0) + 2
                return self.parsed.tokens[idx].value
        return "uniform"

    def if_model_need_filter(self):
        if not self.if_contain_groupby():
            return False
        x = self.get_x()
        gbs = self.get_groupby_value()

        # print("x", x)
        if x[0][0] in gbs:
            return True
        else:
            return False

    def get_query_type(self):
        item = self.parsed.tokens
        if item[0].ttype is DML:
            return "select"
        elif item[0].ttype is DDL and item[0].normalized == "CREATE":
            return "create"
        elif item[0].ttype is DDL and item[0].normalized == "DROP":
            return "drop"
        elif item[0].ttype is Keyword and item[0].normalized == "SET":
            return "set"
        elif item[0].ttype is Keyword and item[0].normalized == "SHOW":
            return "show"

        else:
            warnings.warn("Unexpected SQL:")

    def get_set_variable_value(self):
        item = self.parsed.tokens
        if item[0].ttype is Keyword and item[0].normalized == "SET":
            for comparison in item:
                # print("comparison", comparison)
                if isinstance(comparison, sql.Comparison) or isinstance(comparison, sql.IdentifierList):
                    # print("SQL contain comparison.")
                    splits = comparison.value.split("=")
                    # print("splits[1]", splits[1])
                    if any(i in splits[1] for i in ["'", '"']):
                        # value is a string
                        # print("value is string")
                        splits[1] = splits[1].replace("'", "").replace('"', '')
                        # print("splits[1] BEFORE", splits[1])
                        if splits[1].lower() == "true":
                            splits[1] = True
                        elif splits[1].lower() == "false":
                            splits[1] = False
                        else:
                            pass

                        # print("splits[1]", splits[1])
                    elif "." in splits[1]:
                        splits[1] = float(splits[1])
                    else:
                        splits[1] = int(float(splits[1]))
                    return splits[0], splits[1]
        warnings.warn(
            "error parsing the SQL. Possible solution is set val='True' instead of set val=True")
        return

        # def get_filter(self):
        #     x_between_and = self.get_where_x_and_range()
        #     gbs = self.get_groupby_value()

        #     # print("x_between_and", x_between_and)
        #     if x_between_and[0] not in gbs:
        #         return None
        #     else:
        #         try:
        #             return [float(item) for item in x_between_and[1:]]
        #         except ValueError:
        #             # check if timestamp exists
        #             if "unix_timestamp" in x_between_and[1]:
        #                 # print([unix_timestamp(item.replace("unix_timestamp(", "").replace(")", "").replace("'", "").replace('"', '')) for item in x_between_and[1:]])
        #                 return [unix_timestamp(item.replace("unix_timestamp(", "").replace(")", "").replace("'", "").replace('"', '')) for item in x_between_and[1:]]
        #             else:
        #                 raise ValueError("Error parse SQL.")


if __name__ == "__main__":
    parser = DBEstParser()
    # ---------------------------------------------------------------------------------------------------------------------------------
    # DDL
    # parser.parse(
    #     "create table mdl ( y categorical distinct, x0 real , x2 categorical, x3 categorical) from tbl group by z method uniform size '/data/haha.csv'")

    # print(parser.get_query_type())
    # if parser.if_contain_groupby():
    #     print("yes, group by")
    #     print(parser.get_groupby_value())
    # else:
    #     print("no group by")

    # if parser.if_ddl():
    #     print("ddl")
    #     print(parser.get_ddl_model_name())
    #     print(parser.get_y())
    #     print(parser.get_x())
    #     print(parser.get_from_name())
    #     print(parser.get_sampling_method())
    #     print(parser.get_sampling_ratio())
    #     print(parser.if_model_need_filter())

    # parser.parse(
    #     "select count(y) from t_m where    1 <=x <=2 GROUP BY z1, z2 ,z3 method uniform")  # scale file
    # print(parser.if_contain_scaling_factor())

    # ---------------------------------------------------------------------------------------------------------------------------------
    # DML
    # parser.parse(
    #     "select z, count ( y ) from t_m where x BETWEEN  unix_timestamp('2019-02-28T16:00:00.000Z') and unix_timestamp('2019-03-28T16:00:00.000Z') and 321<X1 < 1123 and x2 = 'HaHaHa' and x3='' and x4<5 GROUP BY z1, z2 ,x method uniform scale data   haha/num.csv  size 23")
    parser.parse(
        "select z, var ( y ) from t_m where  unix_timestamp('2019-02-28T16:00:00.000Z')<=x <=unix_timestamp('2019-03-28T16:00:00.000Z') and 321<X1 < 1123 and x2 = 'HaHaHa' and x3='' and x4<5 GROUP BY z1, z2 ,x method uniform scale data   haha/num.csv  size 23")
    # print(parser.if_contain_scaling_factor())
    if parser.if_contain_groupby():
        print("yes, group by")
        print(parser.get_groupby_value())
    else:
        print("no group by")
    if not parser.if_ddl():
        print("DML")
        print(parser.get_dml_aggregate_function_and_variable())

    # if parser.if_where_exists():
    #     print("where exists!")
    #     # print(parser.get_where_x_and_range())
    #     # print(parser.get_dml_where_categorical_equal_and_range())

    # print("method, ", parser.get_sampling_method())

    # # print("scaling factor ", parser.get_scaling_method())

    # # print(parser.get_where_x_and_range())
    # print(parser.get_dml_where_categorical_equal_and_range())

    # print(parser.get_query_type())

    # print(parser.get_filter())

    # ---------------------------------------------------------------------------------------------------------------------------------
    # set SQL
    # parser.parse("set encoder='gpu'")
    # print(parser.parsed)
    # print(parser.get_query_type())
    # print((parser.get_set_variable_value()))

    # ---------------------------------------------------------------------------------------------------------------------------------
    # drop SQL
    # parser.parse("  drop table haha")
    # print(parser.get_query_type())
    # print(parser.drop_get_model())

    # ---------------------------------------------------------------------------------------------------------------------------------
    # show SQL
    # parser.parse("show tables")
    # print(parser.get_query_type())
