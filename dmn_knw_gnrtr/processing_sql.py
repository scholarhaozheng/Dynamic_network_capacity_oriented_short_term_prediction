import mysql.connector
import datetime

combined_table = False
date_filter_need = False
date_filter_and_combine = True

# 数据库连接参数
host = 'localhost'
user = 'root'
password = 'zxczxc@1234'
database = 'suzhoudata0513'
mother_table ='suzhou202301_all'

# 需要筛选的日期列表
filter_dates = ['2023-01-04', '2023-01-05', '2023-01-06']

# Establish connection
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='zxczxc@1234',
    database='suzhoudata0513',
)

# Check if the connection is successful
print(conn)
print(type(conn))

cursor = conn.cursor()

cursor.execute(f"SELECT * FROM {mother_table} LIMIT 10")
data = cursor.fetchall()

cursor.execute("SHOW TABLES")
tables = cursor.fetchall()
print("Tables in the database:", tables)
cursor.execute(f"DESCRIBE {mother_table}")
table_structure = cursor.fetchall()
print("表结构：")
for col in table_structure:
    print(col)

try:
    # 连接到数据库
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )

    if conn.is_connected():
        print("成功连接到数据库")

    cursor = conn.cursor()

    cursor.execute(f"DESCRIBE {mother_table}")
    table_structure = cursor.fetchall()
    print("表结构：")
    for col in table_structure:
        print(col)

    cursor.execute(f"SELECT DISTINCT data_date FROM {mother_table} LIMIT 10")
    dates = cursor.fetchall()
    print("\n日期样本：", dates)

    if combined_table:
        combined_table_name = "combined_suzhou2023_03_04"
        print(f"\n合并 suzhou2023_03 和 suzhou2023_04 数据，新的表名: {combined_table_name}")

        insert_combined_query = f"""
            INSERT INTO {combined_table_name}
            SELECT * FROM suzhou2023_03
            UNION ALL
            SELECT * FROM suzhou2023_04
            """
        cursor.execute(insert_combined_query)
        conn.commit()
        print(f"表 '{combined_table_name}' 已成功创建，并插入了 suzhou2023_03 和 suzhou2023_04 的数据。")

        # 提取 suzhou2023_03 和 suzhou2023_04 的所有数据并合并为一张表
        cursor.execute(f"DROP TABLE IF EXISTS {combined_table_name}")
        cursor.execute(f"CREATE TABLE {combined_table_name} LIKE suzhou2023_03")

    if date_filter_need:
    # 筛选指定日期的数据并创建新表
        for dt in filter_dates:
            new_table_name = f"{mother_table}_filtered_{dt.replace('-', '_')}"
            print(f"\n处理日期: {dt}, 新表名: {new_table_name}")

            cursor.execute(f"DROP TABLE IF EXISTS {new_table_name}")
            cursor.execute(f"CREATE TABLE {new_table_name} LIKE {mother_table}")

            insert_query = f"""
            INSERT INTO {new_table_name}
            SELECT * FROM {mother_table}
            WHERE data_date = '{dt}'
            """
            cursor.execute(insert_query)
            conn.commit()

            print(f"新表 '{new_table_name}' 已创建，并成功插入 {dt} 的记录。")

    if date_filter_and_combine:
        def extract_filtered_data(cursor, conn, source_table_name, new_table_name, start_date, end_date):
            print(f"\n提取 {source_table_name} 中 {start_date} 至 {end_date} 的数据，新的表名: {new_table_name}")

            drop_table_query = f"DROP TABLE IF EXISTS {new_table_name}"
            cursor.execute(drop_table_query)

            create_table_query = f"CREATE TABLE {new_table_name} LIKE {source_table_name}"
            cursor.execute(create_table_query)

            insert_filtered_query = f"""
                INSERT INTO {new_table_name}
                SELECT * FROM {source_table_name}
                WHERE data_date BETWEEN '{start_date}' AND '{end_date}'
            """
            cursor.execute(insert_filtered_query)

            conn.commit()

            print(f"表 '{new_table_name}' 已成功创建，并插入了 {source_table_name} 中 {start_date} 至 {end_date} 的数据。")


        """extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_05",
            new_table_name="suzhou2023_05_filtered_20230501_20230510",
            start_date="2023-05-01",
            end_date="2023-05-10"
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_05",
            new_table_name="suzhou2023_05_filtered_20230511_20230520",
            start_date="2023-05-11",
            end_date="2023-05-20"
        )"""

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_05",
            new_table_name="suzhou2023_05_filtered_20230521_20230530",
            start_date="2023-05-21",
            end_date="2023-05-30"
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_04",
            new_table_name="suzhou2023_04_filtered_20230401_20230410",
            start_date="2023-04-01",
            end_date="2023-04-10"
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_04",
            new_table_name="suzhou2023_04_filtered_20230411_20230420",
            start_date="2023-04-11",
            end_date="2023-04-20"
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_04",
            new_table_name="suzhou2023_04_filtered_20230421_20230430",
            start_date="2023-04-21",
            end_date="2023-04-30"
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_06",
            new_table_name="suzhou2023_06_filtered_20230601_20230610",
            start_date="2023-06-01",
            end_date="2023-06-10"
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_06",
            new_table_name="suzhou2023_06_filtered_20230611_20230620",
            start_date="2023-06-11",
            end_date="2023-06-20"
        )

        extract_filtered_data(
            cursor=cursor,
            conn=conn,
            source_table_name="suzhou2023_05",
            new_table_name="suzhou2023_05_filtered_20230621_20230630",
            start_date="2023-06-21",
            end_date="2023-06-30"
        )

except mysql.connector.Error as err:
    print("数据库错误:", err)

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("\n数据库连接已关闭。")
