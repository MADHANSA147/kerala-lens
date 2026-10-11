import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from unittest.mock import patch
from app import App

root=tk.Tk()
root.withdraw()
app=App(root)
app.display({'label':'Uncertain','decision':-1,'reason':'Insufficient visual evidence.','mode':'Test fixture',
             'candidates':[{'latitude':10.,'longitude':76.,'in_kerala':True}]})
root.update_idletasks()
assert app.heading.cget('text') == 'Uncertain'
assert len(app.table.get_children()) == 1
with patch('app.messagebox.showinfo') as message:
    app.accuracy_report()
    assert message.call_count == 1
    assert 'held-out towns' in message.call_args.args[1]
app.events.put(('error','An invalid image was rejected.'))
app.poll()
assert app.heading.cget('text') == 'Could not analyse'
root.destroy()
print('Desktop interface smoke checks passed.')
