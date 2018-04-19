#!/usr/bin/python2.7
# -*- coding: latin-1 -*-

import pandas as pd
import ezodf as ez
import os

from numpy import zeros

class File():
    def __init__(self, filename, verbose=False, **kwargs):
        print("\x1b[1;37;43mCode OK only for one sheet size Spreadsheets \x1b[0m")
        
        self.new_col = False
        self.new_row = False
        
#        if "new_cols" in kwargs.keys():
#            self.new_col = True
#            self.param_col = kwargs["param_col"] #tuple ou list
#            
#            
#        if "new_rows" in kwargs.keys():
#            self.new_row = True
#            self.param_row = kwargs["param_row"]

        
        if os.path.exists(filename) == False :
            if os.path.splitext(os.path.abspath(filename))[1] != "ods" :
                filename = os.path.splitext(os.path.abspath(filename))[0] + ".ods"
                
            doc = ez.newdoc(os.path.abspath(filename))
        
        else :
            doc = ez.opendoc(filename)
        
        self.doc = doc
        self.kwargs = kwargs
        self.filename = filename
#------------------------------------------------------------------------------
    def read_file(self):
        print("Spreadsheet %s contains %d sheet(s)." % (self.filename, len(self.doc.sheets)))
        for sheet in self.doc.sheets:
            print("-"*40)
            print("Size of Sheet : (rows=%d, cols=%d)" % (sheet.nrows(), sheet.ncols()) )
        
        # convert the first sheet to a pandas.DataFrame
        sheet_dict = {}
        for j, sheet in enumerate(self.doc.sheets) :
            for i, row in enumerate(sheet.rows()):
                # row is a list of cells
                # assume the header is on the first row
                if i == 0:
                    # columns as lists in a dictionary
                    sheet_dict = {cell.value:[] for cell in row}
                    # create index for the column headers
                    col_index = {j:cell.value for j, cell in enumerate(row)}
                    continue
                    
                for j, cell in enumerate(row):
                    # use header instead of column index
                    sheet_dict[col_index[j]].append(cell.value)
                    
            # and convert to a DataFrame
            df = pd.DataFrame(sheet_dict)
        
        self.key_to_num = {}
        
        for j, c in enumerate(df.columns) :
            self.key_to_num[c] = j
        
        self.df = df
#------------------------------------------------------------------------------
    def write_in_file(self, row_data) :
        for item in row_data.iteritems():
            if type(item[1]) is None :
                item[0] = 'None'
                
        sheet = self.doc.sheets[0]
        row = {}

        for item in row_data.iteritems() :
            print ("Numero d'indice modifier par {} : {}".format(item[0], self.key_to_num[item[0]]))
            print ("valeur : {}".format(item[1]))
            row["%03d" % self.key_to_num[item[0]]] = item[1]
        
        self.doc.sheets[0].append_rows(1)
        
        for i in range(len(row_data.keys())) :
            try : 
                sheet[sheet.nrows() -1 , i].set_value(row["%03d" % i])
            except AttributeError :
                sheet[sheet.nrows()-1, :].set_value(str(row["%03d" % i]))
        
        self.doc.save()
#------------------------------------------------------------------------------
if __name__ == '__main__' :
    f = File("./some_odf_spreadsheet.ods")
    f.read_file()
    
    
