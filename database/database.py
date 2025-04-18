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
        """Get the value of a variable by its name"""
        if datatype == "offset":
            if id == "0":
                return self.offset0
            elif id == "1":
                return self.offset1
            elif id == "2":
                return self.offset2
            elif id == "3":
                return self.offset3
            elif id == "4":
                return self.offset4    
            elif id == "5":
                return self.offset5
            else:
                raise ValueError(f"Unknown offset ID: {id}")
        else:
            raise ValueError(f"Unknown datatype: {datatype}")
    
    def set(self, datatype, data, id):
        """Set the value of a variable by its name"""
        if datatype == "offset":
            if id == "0":
                self.offset0 = data
            elif id == "1":
                self.offset1 = data
            elif id == "2":
                self.offset2 = data
            elif id == "3":
                self.offset3 = data
            elif id == "4":
                self.offset4 = data
            elif id == "5":
                self.offset5 = data
                print(f"Setting offset5 to {data}")
        else:
            raise ValueError(f"Unknown datatype: {datatype}")

    def __str__(self):
        """String representation of the database"""
        return f"Database(offset0={self.offset0}, offset1={self.offset1}, offset2={self.offset2}, offset3={self.offset3}, offset4={self.offset4}, offset5={self.offset5})"
    
    def __repr__(self):
        """String representation of the database"""
        return self.__str__()
            

