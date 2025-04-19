#Baza danych do przechowywania zmiennych

class Database:
    def __init__(self):
        self.offset0 = 0
        self.offset1 = 0
        self.offset2 = 0
        self.offset3 = 0
        self.offset4 = 0
        self.offset5 = 0

    def get(self, datatype, id):
        if datatype == "offset":
            return getattr(self, f"offset{id}", 0)
        return None
    
    def set(self, datatype, data, id):
        if datatype == "offset":
            setattr(self, f"offset{id}", data)

    def __str__(self):
        """String representation of the database"""
        return f"Database(offset0={self.offset0}, offset1={self.offset1}, offset2={self.offset2}, offset3={self.offset3}, offset4={self.offset4}, offset5={self.offset5})"
    
    def __repr__(self):
        """String representation of the database"""
        return self.__str__()
            

