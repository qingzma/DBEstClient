class Slave:
    """ define slaves in socket programing.
    """

    def __init__(self, host_port: str):
        if ":" not in host_port:
            raise ValueError(
                "The format of slave definition is incorrect, should be <host>:<port>")
        splits = host_port.lstrip().rstrip().split(":")
        self.host = splits[0]
        try:
            self.port = int(splits[1])
        except ValueError:
            raise ValueError("The port should be an integer.")

    def to_string(self) -> str:
        """ convert to string

        Returns:
            [str]: description of the slave.
        """
        return "<"+self.host+">:<"+str(self.port)+">"


class Slaves:
    def __init__(self):
        self.container = {}

    def add(self, slave: Slave):
        self.container[slave.to_string()] = slave

    def delete(self, host: str, port: int):
        slave = Slave(host+":"+str(port))
        if slave.to_string() in self.container:
            del self.container[slave.to_string()]
        else:
            print("Host does not exists in the slaves file, not deleted.")

    def to_string(self):
        slaves_str = ""
        for key in self.container:
            slaves_str = slaves_str + self.container[key].to_string() + ","
        if slaves_str:
            slaves_str = slaves_str[:-1]
        return slaves_str

    def is_empty(self):
        if self.container:
            return False
        else:
            return True

    def get(self):
        return self.container

    def size(self):
        return len(self.container)


class UseCols:
    def __init__(self, usecols):
        # {'y': ['ss_sales_price', 'real', None], 'x_continous': ['ss_sold_date_sk'], 'x_categorical': ['ss_coupon_amt'], 'gb': ['ss_store_sk']}
        self.usecols = usecols
        self.continous_cols = None
        self.categorical_cols = None
        self.overlap_cols = []

    def get_continous_and_categorical_cols(self):
        columns_as_continous = [self.usecols['y'][0]]
        if self.usecols['x_continous']:
            columns_as_continous = columns_as_continous + \
                self.usecols['x_continous']

        columns_as_categorical = self.usecols["x_categorical"] + \
            self.usecols["gb"]
        # print("columns_as_continous", columns_as_continous)
        # print("columns_as_categorical", columns_as_categorical)

        # remove the x column in continous if the column also appear in group by
        for col_item in columns_as_continous:
            if col_item in columns_as_categorical:
                # print("col_item", col_item)
                columns_as_continous.remove(col_item)
                self.overlap_cols.append(col_item)

        self.continous_cols = columns_as_continous
        self.categorical_cols = columns_as_categorical
        return self.continous_cols, self.categorical_cols, self.overlap_cols

    def get_gb_x_y_cols_for_one_model(self):
        gb = self.usecols["gb"] + self.usecols['x_categorical']
        x = self.usecols["x_continous"]
        y = [self.usecols['y'][0]]
        return gb, x, y
