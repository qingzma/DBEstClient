import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function
from sqlparse.tokens import Keyword, DML, DDL


class DBEstParser:
    def __init__(self):
        self.query = ""
        self.parsed = None

    def parse(self, query):
        self.query = query
        self.parsed = sqlparse.parse(self.query)[0]
        # return self.parsed

    def if_nested_query(self):
        if not self.parsed.is_group:
            return False
        for item in self.parsed.tokens:
            if item.ttype is DML and item.value.upper() == 'SELECT':
                return True
        return False

    def if_contain_groupby(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "group by":
                return True
        return False

    def get_groupby_value(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "group by":
                idx = self.parsed.token_index(item,0) + 2
                return self.parsed.tokens[idx].value

    def if_ddl(self):
        for item in self.parsed.tokens:
            if item.ttype is DDL and item.value.lower() == "create":
                return True
        return False

    def get_model_name(self):
        for item in self.parsed.tokens:
            if item.ttype is  None and "(" in item.value.lower():
                return item.tokens[0].value

    def get_y(self):
        for item in self.parsed.tokens:
            if item.ttype is  None and "(" in item.value.lower():
                return item.tokens[1].tokens[1].value,item.tokens[1].tokens[3].value

    def get_x(self):
        for item in self.parsed.tokens:
            if item.ttype is  None and "(" in item.value.lower():
                return item.tokens[1].tokens[6].value,item.tokens[1].tokens[8].value

    def get_table_name(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "from":
                idx = self.parsed.token_index(item, 0) + 2
                return self.parsed.tokens[idx].value

    def get_sampling_ratio(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "size":
                idx = self.parsed.token_index(item,0) + 2
                return self.parsed.tokens[idx].value
        return 0.01

    def get_sampling_method(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "method":
                idx = self.parsed.token_index(item,0) + 2
                return self.parsed.tokens[idx].value
        return "uniform"



if __name__=="__main__":
    parser = DBEstParser()
    parser.parse("create table mdl(y real, x real) from tbl group by z method uniform size 0.1 ")

    if parser.if_contain_groupby():
        print("yes, group by")
        print(parser.get_groupby_value())
    else:
        print("no group by")

    if parser.if_ddl():
        print("ddl")
        print(parser.get_model_name())
        print(parser.get_y())
        print(parser.get_x())
        print(parser.get_table_name())
        print(parser.get_sampling_method())
        print(parser.get_sampling_ratio())

