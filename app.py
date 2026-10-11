"""Offline desktop interface. No web server or remote image uploads."""
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from kerala_geo.core import ROOT, read_image


class App:
    def __init__(self, root):
        self.root = root
        root.title('Kerala Lens | Offline image geolocation')
        root.geometry('1080x780')
        root.minsize(920, 680)
        root.configure(bg='#f3f5f0')
        self.events = queue.Queue()
        self.predictor = None
        self.result = None
        self.source = None
        self.busy = False
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f3f5f0')
        style.configure('TLabel', background='#f3f5f0', foreground='#213b34', font=('Segoe UI',11))
        style.configure('Title.TLabel', font=('Segoe UI',28,'bold'))
        style.configure('Sub.TLabel', foreground='#5b6f66', font=('Segoe UI',10))
        style.configure('TButton', font=('Segoe UI',11), padding=10)
        style.configure('Go.TButton', foreground='white', background='#176b53')
        style.configure('Treeview', rowheight=30, font=('Segoe UI',10))
        style.configure('Treeview.Heading', font=('Segoe UI',10,'bold'))
        outer = ttk.Frame(root, padding=28)
        outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='KERALA LENS',style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer,text='Visual location estimates. On your computer. Offline.',style='Sub.TLabel').pack(anchor='w',pady=(4,20))
        body = ttk.Frame(outer)
        body.pack(fill='both',expand=True)
        left, right = ttk.Frame(body), ttk.Frame(body,padding=(24,0,0,0))
        left.pack(side='left',fill='y')
        right.pack(side='left',fill='both',expand=True)
        self.canvas = tk.Canvas(left,width=350,height=300,bg='#e4eae2',highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(175,150,text='Choose a photo to begin',fill='#5b6f66',font=('Segoe UI',12))
        self.filename = ttk.Label(left,text='JPEG, PNG, or WebP',style='Sub.TLabel',wraplength=350)
        self.filename.pack(anchor='w',pady=(10,16))
        self.choose_button = ttk.Button(left,text='Choose image',command=self.choose)
        self.choose_button.pack(fill='x')
        self.analyse_button = ttk.Button(left,text='Analyse photo',style='Go.TButton',command=self.analyse,state='disabled')
        self.analyse_button.pack(fill='x',pady=8)
        self.save_button = ttk.Button(left,text='Save result',command=self.save,state='disabled')
        self.save_button.pack(fill='x')
        ttk.Button(left,text='View accuracy report',command=self.accuracy_report).pack(fill='x',pady=(8,0))
        ttk.Label(left,text='Your photo stays on this computer.\nGPS metadata is not used for prediction.',style='Sub.TLabel',wraplength=350).pack(anchor='w',pady=20)
        self.status = ttk.Label(outer,text='Ready. First analysis loads the model and prepares its location index.',style='Sub.TLabel',wraplength=960)
        self.status.pack(anchor='w',pady=(20,0))
        self.heading = ttk.Label(right,text='Where might this be?',font=('Segoe UI',23,'bold'))
        self.heading.pack(anchor='w')
        self.reason = ttk.Label(right,text='The model estimates possible locations from visual clues. Some photos do not contain enough information.',wraplength=570)
        self.reason.pack(anchor='w',pady=(12,8))
        self.mode = ttk.Label(right,text='Accuracy has not been established for your photos.',style='Sub.TLabel',wraplength=570)
        self.mode.pack(anchor='w',pady=(0,20))
        ttk.Label(right,text='Approximate location candidates',font=('Segoe UI',12,'bold')).pack(anchor='w',pady=(0,10))
        self.table = ttk.Treeview(right,columns=('rank','lat','lon','region'),show='headings',height=10)
        for name,title,width in [('rank','#',35),('lat','Latitude',100),('lon','Longitude',100),('region','Boundary check',170)]:
            self.table.heading(name,text=title)
            self.table.column(name,width=width,anchor='w')
        self.table.pack(fill='x')
        ttk.Label(right,text='Coordinates are estimates, not verified capture locations. Similarity scores are not accuracy percentages.',style='Sub.TLabel',wraplength=570).pack(anchor='w',pady=15)
        root.after(100,self.poll)

    def choose(self):
        path = filedialog.askopenfilename(title='Choose an image',filetypes=[('Photos','*.jpg *.jpeg *.png *.webp'),('All files','*.*')])
        if not path:
            return
        try:
            image = read_image(path)
            image.thumbnail((350,300))
            self.preview = ImageTk.PhotoImage(image)
            self.canvas.delete('all')
            self.canvas.create_image(175,150,image=self.preview)
        except ValueError as exc:
            messagebox.showerror('Cannot open image',str(exc))
            return
        self.source = path
        self.filename.config(text=Path(path).name)
        self.result = None
        self.table.delete(*self.table.get_children())
        self.heading.config(text='Ready to analyse')
        self.reason.config(text='Choose Analyse photo to estimate likely locations.')
        self.analyse_button.config(state='normal')
        self.save_button.config(state='disabled')

    def analyse(self):
        if self.busy or not self.source:
            return
        self.busy = True
        self.result = None
        self.choose_button.config(state='disabled')
        self.analyse_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.heading.config(text='Analysing...')
        self.status.config(text='Loading local model. This can take a minute on the first run.')
        source = self.source
        def work():
            try:
                from kerala_geo.model import Predictor
                if self.predictor is None:
                    self.predictor = Predictor(progress=lambda m:self.events.put(('status',m)))
                self.events.put(('result',self.predictor.predict(source)))
            except Exception as exc:
                self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,value = self.events.get_nowait()
                if kind == 'status':
                    self.status.config(text=value)
                    continue
                self.busy = False
                self.choose_button.config(state='normal')
                self.analyse_button.config(state='normal')
                if kind == 'error':
                    self.heading.config(text='Could not analyse')
                    self.reason.config(text=value)
                    self.status.config(text='Check Setup.cmd and the README for help.')
                else:
                    self.display(value)
        except queue.Empty:
            pass
        self.root.after(100,self.poll)

    def display(self,result):
        self.result = result
        prefix = 'Estimate: ' if result['decision'] != -1 else ''
        self.heading.config(text=prefix+result['label'])
        self.reason.config(text=result['reason'])
        self.mode.config(text=result['mode'])
        self.table.delete(*self.table.get_children())
        for i,candidate in enumerate(result['candidates'],1):
            self.table.insert('', 'end',values=(i,f"{candidate['latitude']:.3f}",f"{candidate['longitude']:.3f}",'Kerala' if candidate['in_kerala'] else 'Outside Kerala'))
        self.status.config(text='Analysis complete. No image was uploaded.')
        self.save_button.config(state='normal')

    def save(self):
        path = filedialog.asksaveasfilename(title='Save result',defaultextension='.json',initialfile='location-result.json',filetypes=[('JSON result','*.json')])
        if path:
            try:
                Path(path).write_text(json.dumps(self.result,indent=2),encoding='utf-8')
            except OSError as exc:
                messagebox.showerror('Cannot save result',str(exc))

    def accuracy_report(self):
        try:
            report=json.loads((ROOT/'trained/report.json').read_text(encoding='utf-8'))
            forced=report['held_out_test_forced_binary']
            conservative=report['held_out_test']
            baseline=report['held_out_geoclip_top1_boundary_baseline']
            text=(f"Test photos: {forced['images']} from held-out towns\n\n"
                  f"Exploratory binary accuracy: {forced['accuracy_on_accepted']:.1%}\n"
                  f"GeoCLIP coordinate baseline: {baseline['accuracy_on_accepted']:.1%}\n\n"
                  f"Conservative mode answered {conservative['accepted']} photos and marked {conservative['uncertain']} uncertain.\n\n"
                  'This is a small public-photo pilot, not a guarantee of accuracy on everyday photos. '
                  'The project is not ready for dependable location verification.')
            messagebox.showinfo('Measured pilot performance',text)
        except (OSError,ValueError,KeyError):
            messagebox.showinfo('Accuracy not measured','No local training report is available yet.')


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
