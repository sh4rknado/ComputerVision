import mysql.connector
import traceback
from Helper.Colors import Colors


class SQLHelpers:
    def __init__(self, user='zmuser', passwd='zmpass', host='localhost', database='zm'):
        try:
            self._db = mysql.connector.connect(host=host, user=user, passwd=passwd, database=database)
        except mysql.connector.Error as err:
            Colors.print_error("\n[ERROR] Could not connect to " + host + "\n")
            traceback.print_exc()
            exit(1)

    # Events Manager
    def delete_events(self, ID):
        SQL = "DELETE FROM Events WHERE Id=" + ID
        results = self.execute(SQL)
        self._db.commit()
        return results

    def delete_frames(self, EventID,FramID):
        SQL = "DELETE FROM * WHERE EventId=" + EventID + " AND FrameId=" + FramID
        results = self.execute(SQL)
        self._db.commit()
        return results

    def get_events(self, ID):
        SQL = "SELECT * FROM Events WHERE Id=" + ID
        results = self.execute(SQL)
        self._db.commit()
        return results

    def get_all_events(self):
        SQL = "SELECT * FROM Events"
        results = self.execute(SQL)
        self._db.commit()
        return results

    def execute(self, sql):
        mycursor = self._db.cursor()
        mycursor.execute(sql)
        results = mycursor.fetchall()
        mycursor.close()
        del mycursor
        return results


if __name__ == "__main__":
    sql = SQLHelpers()
    result = sql.execute("select * from Frames Where EventId=10280")
    for x in result:
        print(x)

    result = sql.get_events("1")
    for x in result:
        print(x)
