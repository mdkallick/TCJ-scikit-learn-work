from tkinter import *
from tkinter import filedialog
# import tkinter
from arap_clustering import gui_func

class GUI:

    def __init__(self, master):
        frame = Frame(master)
        frame.pack()

        self.filename = None
        self.out = None

        text_height=2
        self.filename_disp = Text(frame, height=text_height)
        self.out_disp = Text(frame, height=text_height)

        self.pick_file = Button(
            frame, text="Pick Data File", fg="red", command=self.get_filename
        )
        self.pick_file.pack(side=TOP)
        self.filename_disp.pack(side=TOP)

        self.pick_out = Button(
            frame, text="Pick Output File", fg="red", command=self.get_out_filename
        )
        self.pick_out.pack(side=TOP)
        self.out_disp.pack(side=TOP)

        self.quit = Button(
            frame, text="Quit", fg="red", command=frame.quit
        )
        self.quit.pack(side=LEFT)

        self.run_clustering = Button(
            frame, text="Cluster", fg="red", command=self.run_clustering
        )
        self.run_clustering.pack(side=RIGHT)
        self.run_clustering['state'] = DISABLED

    def get_filename(self):
        selected = filedialog.askopenfilename(
                        # initialdir="C:\\",
                        title = "Select data file",
                        filetypes = (("csv files","*.csv"),("all files","*.*"))
                        )
        self.filename = selected
        self.filename_disp.delete(1., END)
        self.filename_disp.insert(1., selected)

        if(self.out != None):
            self.run_clustering['state'] = 'normal'

    def get_out_filename(self):
        selected = filedialog.asksaveasfile(
                        # initialdir="C:\\",
                        title = "Select file to write to",
                        filetypes = (("csv files","*.csv"),("all files","*.*"))
                        )
        self.out = selected
        # print("selected", selected.name)
        self.out_disp.delete(1.)
        self.out_disp.insert(1., selected.name)



        if(self.filename != None):
            self.run_clustering['state'] = 'normal'

    def run_clustering(self):
        gui_func('', self.filename, self.out)

if __name__ == '__main__':
    root = Tk()

    gui = GUI(root)

    root.mainloop()
    root.destroy()
