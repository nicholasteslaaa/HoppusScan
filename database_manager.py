import sqlite3
import pandas as pd

class db_manager:
    def __init__(self) -> None:
        self.connection = sqlite3.connect('data.db',check_same_thread=False)

        self.cursor = self.connection.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chairs (
                bbox TEXT,
                timer FLOAT
            )
        ''')
    
        
    def pop_data(self,box):
        str_box = " ".join([str(i) for i in box])
        self.cursor.execute(f"DELETE FROM chairs WHERE bbox = '{str_box}'")
        self.connection.commit()
        
    def update_timer(self,box,time):
        str_box = " ".join([str(i) for i in box])
        self.cursor.execute(f"UPDATE chairs SET timer = ? WHERE bbox = ?",(time,str_box))
        self.connection.commit()
            
    # Inside db_manager class
    def insert_data(self, box, timer):
        str_box = " ".join([str(i) for i in box])
        self.cursor.execute("INSERT INTO chairs (bbox, timer) VALUES (?, ?)", (str_box, timer))
        self.connection.commit()

    def get_all_data_as_dictionary(self):
        df = pd.read_sql_query("SELECT * FROM chairs", self.connection)
        if df.empty:
            return []
            
        data_list = df.to_dict(orient='records')
        for data in data_list:
            # Convert "10 20 100 200" back to (10, 20, 100, 200)
            data["bbox"] = tuple(map(int, data["bbox"].split()))
            data["frame"] = None
        return data_list
    
    def close(self):
        self.connection.close()
        
if __name__ == "__main__":
    db = db_manager()
    db.update_timer((1,2,3,4),0)
    
