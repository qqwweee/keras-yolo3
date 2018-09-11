import sqlite3
import os.path
import pytz
from os import listdir, getcwd
from IPython.core.display import Image 
from datetime import datetime, date
from pytz import timezone
import sys
import traceback #add
import io, base64 #add
import os #add
import errno #add

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

#Создать или открыть базу данных
def create_or_open_db(db_file):
    
    #os.path.exists(path) - возвращает True, если path указывает на существующий путь или дескриптор открытого файла
    db_is_new = not os.path.exists(db_file)
    #создать связь с базой данных
    conn = sqlite3.connect(db_file) #,detect_types=sqlite3.PARSE_DECLTYPES
    if db_is_new:
        print('Creating schema main')
        conn.execute("PRAGMA foreign_keys = ON;")
        sql = '''create table if not exists PICTURES(
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        PICTURE BLOB,
        [timestamp] timestamp);'''
        conn.execute(sql) # shortcut for conn.cursor().execute(sql)
        sql2 = "create table if not exists CLASSES( ID_CLASS INTEGER PRIMARY KEY AUTOINCREMENT, CLASS TEXT, UNIQUE (CLASS) ON CONFLICT ABORT);" #CLASS TEXT, UNIQUE (CLASS) ON CONFLICT IGNORE
        conn.execute(sql2) # shortcut for conn.cursor().execute(sql)
        sql3 = '''create table if not exists CHILD_PICTURES( PICTURES_ID INTEGER, CLASSES_ID_CLASS INTEGER,
        PERCENT REAL, CHILD_PICTURE BLOB, FACE_ID TEXT, FACE_ACCURACY REAL, FOREIGN KEY(PICTURES_ID) REFERENCES PICTURES(ID) ON UPDATE CASCADE ON DELETE CASCADE,
        FOREIGN KEY(CLASSES_ID_CLASS) REFERENCES CLASSES(ID_CLASS) ON UPDATE CASCADE ON DELETE CASCADE);'''
        conn.execute(sql3) # shortcut for conn.cursor().execute(sql)
    else:
        print('Schema exists main\n')
    return conn

#Вставить (изображение) в базу данных table PICTURES
def insert_picture(conn, picture_file):
    try:
        #чтение в двоичном режим
        with io.BytesIO(picture_file) as input_file:
            ablob = input_file.read()
    except Exception as e:
        
        print('Ошибка main:\n', traceback.format_exc())
    else:
        sql = '''INSERT INTO PICTURES
        (PICTURE, timestamp)
        VALUES(?, ?);'''
        conn.execute(sql,[sqlite3.Binary(ablob), datetime.strftime(datetime.now(pytz.timezone('Europe/Moscow')), "%d.%m.%Y %H:%M:%S")])
        conn.commit()

        #sql = "SELECT PICTURE " +\
        #"FROM PICTURES WHERE id = :id"

        #param = {'id': picture_id}
        #cursor.execute(sql, param)
        print('The image was obtained (main)\n')

#Удаление записи по id table PICTURES
def delete_picture(conn, picture_id):
    try:
        sql = "DELETE FROM PICTURES WHERE id = :id"
        param = {'id': picture_id}

        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(sql, param)
        conn.commit()
    
    except sqlite3.Error:
        print("Error sqlite3 delete_picture")
        return 1
    
    else:
        return 0

#Извлечение изображения table PICTURES
def extract_picture(cursor, picture_file, picture_id):

    make_sure_path_exists(picture_file)
    pic = os.path.abspath(picture_file)
    #os.path.exists(path) - возвращает True, если path указывает на существующий путь или дескриптор открытого файла
    if os.path.exists(pic):
        if os.path.isdir(pic):
            pic = pic + "/frame_" + str(picture_id) + ".jpeg"
            print("Extract picture completed successfully")
        else:
            print("Directory not found")
            return 1
    else:
        print("Path incorrect")
        return 1

    try:
        #PICTURE
        sql = "SELECT PICTURE " +\
           "FROM PICTURES WHERE id = :id"
        
        param = {'id': picture_id}
        cursor.execute(sql, param)
    
    except sqlite3.Error:
        print("Error sqlite3 extract_picture")
        return 1
    
    else:    
        #.fetchone() метод возвращает одну запись в виде кортежа, если записей больше нет, тогда он возвращает None
        record = cursor.fetchone()
        if record != None:
            ablob, *_ = record

            with open(pic, 'wb') as output_file:
                output_file.write(base64.b64decode(ablob))
        else:
            print("Error id not found")
            return 1

#Извлечение изображения table CHILD_PICTURES
def child_extract_picture(cursor, picture_file_child, picture_id):

    #Создать папку, если её не существует
    make_sure_path_exists(picture_file_child)
    #Полный путь до picture_file_child
    pic_child = os.path.abspath(picture_file_child)

    try:
        #PICTURE
        sql = '''SELECT CHILD_PICTURES.rowid, CHILD_PICTURES.CHILD_PICTURE, CHILD_PICTURES.PERCENT, CLASSES.CLASS
            FROM CHILD_PICTURES 
            INNER JOIN CLASSES
            ON CHILD_PICTURES.CLASSES_ID_CLASS = CLASSES.ID_CLASS 
            WHERE CHILD_PICTURES.PICTURES_ID = :id;'''

        param = {'id': picture_id}
        cursor.execute(sql, param)
    
    except sqlite3.Error:
        print("Error sqlite3 child_extract_picture")
        return 1
#TODO: доделать всё ниже    
    else:
        return cursor.fetchall()

        # if records != None:
        #     for record in records:
        #         rowid, ablob, PERCENT, CLASSES_ID_CLASS, *_ = record

        #         #os.path.exists(path) - возвращает True, если path указывает на существующий путь или дескриптор открытого файла
        #         if os.path.exists(pic_child):
        #             if os.path.isdir(pic_child):
        #                 print(CLASSES_ID_CLASS)
        #                 el_pic_child = pic_child + "/frame_" + str(picture_id) + "(" + str(rowid) + ")" + "_class_" + str(CLASSES_ID_CLASS) + "_" + str(PERCENT) + ".png"
        #                 with open(el_pic_child, 'wb') as child_output_file:
        #                     child_output_file.write(base64.b64decode(ablob))
        #                     print("Extract picture_child completed successfully")
        #             else:
        #                 print("Directory not found")
        #                 return 1
        #         else:
        #             print("Path incorrect")
        #             return 1
        # else:
        #     print("Error id not found")
        #     return 1

#Вставить (изображение) в базу данных table CHILD_PICTURES
def child_insert_picture(conn, class_name, percent, picture_file_child, face_id='', face_accuracy=0):
    try:
        #чтение в двоичном режим
        with io.BytesIO(picture_file_child) as input_file:
            ablob = input_file.read()
    except Exception as e:
        
        print('Ошибка:\n', traceback.format_exc())
    else:
        sql = '''INSERT INTO CHILD_PICTURES
        (PICTURES_ID, CLASSES_ID_CLASS, PERCENT, CHILD_PICTURE, FACE_ID, FACE_ACCURACY) 
        VALUES((SELECT ID FROM PICTURES ORDER BY ID DESC LIMIT 1), (SELECT ID_CLASS FROM CLASSES WHERE ?=CLASSES.CLASS), ?, ?, ?, ?)'''
        conn.execute(sql,[class_name, percent, sqlite3.Binary(ablob), face_id, face_accuracy])
        conn.commit()
        print('The image was obtained (child)\n')

#---------------Функции для работы с базой данных------------------------

#Создать или открыть базу данных
#def create_or_open_db(db_file)

#Добавление новой записи table PICTURES
#Принимает имя базы данных(база данных там же, где этот файл) и путь к картинке
def add_record(db_name, picture_file):

    #Открываем базу данных
    conn = create_or_open_db(db_name)

    #Добавляем новую запись	
    insert_picture(conn, picture_file)

    #Закрываем базу данных
    conn.close()


#Добавление новой записи table CHILD_PICTURES
#Принимает имя базы данных(база данных там же, где этот файл) и путь к картинке
def add_record_child(db_name, class_name, percent, picture_file_child, face_id='', face_accuracy=0):

    #Открываем базу данных
    conn = create_or_open_db(db_name)

    #Добавляем новую запись	
    child_insert_picture(conn, class_name, percent, picture_file_child, face_id, face_accuracy)

    #Закрываем базу данных
    conn.close()

#Удаление записи по id table PICTURES
def del_record(db_name, id_record):

    create_or_open_db(db_name)
    
    #Открываем базу данных
    conn = create_or_open_db(db_name)

    #Удаляем запись
    delete_picture(conn, id_record)

    #Закрываем базу данных
    conn.close()

#Извлечение записи по id table PICTURES
def extr_record(db_name, picture_file, id_record):
    conn = create_or_open_db(db_name)
    cur = conn.cursor()
    extract_picture(cur, picture_file, id_record)
    cur.close()
    conn.close()

#Извлечение записи по id table CHILD_PICTURES
def child_extr_record(db_name, picture_file_child, id_record):
    conn = create_or_open_db(db_name)
    cur = conn.cursor()
    result = child_extract_picture(cur, picture_file_child, id_record)
    cur.close()
    conn.close()
    return result

#Удаление всех записей из базы данных
def del_all(db_name):
    conn = create_or_open_db(db_name)
    conn.execute("DELETE FROM CLASSES")
    conn.execute("DELETE FROM CHILD_PICTURES")
    conn.execute("DELETE FROM PICTURES")
    conn.commit()
    conn.close()

#Добавление новой записи table CLASSES
def add_record_class(db_name, class_name):
    
    #Открываем базу данных
    conn = create_or_open_db(db_name)

    try:
        sql = '''INSERT INTO CLASSES (CLASS) VALUES(?);'''
        conn.execute(sql, [class_name])
    except Exception as e:
        print("This class already exists\n")
    else:
        conn.commit()
        print("New class successfully added\n")

    #Закрываем базу данных
    conn.close()

#Examples
db = 'test.db'

# del_all(db)


# from urllib.parse import urlencode
# from urllib.request import Request, urlopen

# url = 'https://api.evision.tech/predict/2.0' # Set destination URL here
# post_fields = {
#   "key": "",
#   "minAccuracy": 30,
#   "type": "check",
#   "headers": {
#     'Content-Type': "application/json"
#   },
#   "image": ""
# }

# request = Request(url, urlencode(post_fields).encode())
# json = urlopen(request).read().decode()
# print(json)



# # Запрос в бд изображения
# result = child_extr_record(db, './img_from_bd', 8)

# # print(result)

# if result != None:
#     for record in result:
#         rowid, ablob, PERCENT, CLASSES_ID_CLASS, *_ = record
#         # print(ablob)
#         # child_output_file.write(base64.b64decode(ablob))
#         if CLASSES_ID_CLASS == 'person':
#             from urllib.parse import urlencode
#             from urllib.request import Request, urlopen

#             url = 'https://api.evision.tech/predict/2.0' # Set destination URL here
#             post_fields = {
#                 "key": "",
#                 "minAccuracy": 1,
#                 "type": "check",
#                 "headers": {
#                     'Content-Type': "application/json"
#                 },
#                 "image": "data:image/jpeg;base64," + str(ablob)[2:]
#             }

#             # print(str(ablob)[2:100])
#             print(post_fields['image'][0:100])

#             # print(post_fields)

#             request = Request(url, urlencode(post_fields).encode())
#             json = urlopen(request).read().decode()
#             print(json)





            # print(ablob)
            # print(ablob)
        # print(PERCENT)
        
        # print(ablob)
        # print(ablob)
        # print(ablob)

# print( cursor )



# print(test_data)
# del_all(db)
# add_record(db, 'test_image.jpeg')
# del_record(db, 2)
# extr_record('tatoo.db', 'pic1.png', 1)

#db = 'test.db'
#extr_record(db, './qwerty/FRAME/', 4)

#db = 'test.db'
#extr_record(db, './qwerty/FRAME/', 3)
#child_extr_record(db, './qwerty/OBJECTS/', 1)