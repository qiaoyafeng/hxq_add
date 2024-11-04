import numpy as np
import pymysql
import pandas as pd
from config import settings

mysql_host = settings.DB_IP
db_user = settings.DB_USERNAME
db_pass = settings.DB_PASSWORD
db_name = settings.DB_NAME


def query_sql(sql):
    db = pymysql.connect(
        host=mysql_host, port=3306, user=db_user, passwd=db_pass, db=db_name
    )
    cursor = db.cursor()
    cursor.execute(sql)
    cols = [column[0] for column in cursor.description]
    results = cursor.fetchall()
    df = pd.DataFrame(list(results), columns=cols)
    return df.replace([np.nan], [None], regex=False).to_dict("records")


def query_one(sql, model):
    data = query_sql(sql)
    model = model()
    for key, value in data[0].items():
        setattr(model, key, value)
    return model


def query_list(sql, model):
    res = []
    data = query_sql(sql)
    for d in data:
        new_model = model()
        for key, value in d.items():
            setattr(new_model, key, value)
        res.append(new_model)
    return res


def build_update(info, table):
    sql = f"update {table} set"
    for key, value in info.items():
        if key not in ["id"]:
            sql = sql + f" {key}='{value}',"
    sql = sql[: len(sql) - 1] + f"where id = {info['id']}"
    return sql


def build_create(info, table):
    col = ""
    val = ""
    for key, value in info.items():
        if value or value == 0:
            val = val + f" '{value}',"
            col = col + f" {key},"

    sql = f"insert into {table} ({col[:len(col) - 1]})values({val[:len(val) - 1]})"
    return sql


def update_sql(sql):
    db = pymysql.connect(
        host=mysql_host, port=3306, user=db_user, passwd=db_pass, db=db_name
    )
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
