import tkinter
from setup import model

window=tkinter.Tk()
window.title('Vector Space Model')
window.attributes('-fullscreen', True)
window.configure(bg='#ffffff')


query_input=tkinter.StringVar()
alpha_input=tkinter.StringVar()
initial_content='1) Type in any free text query of form eg: sent due,walked quickly,face of agony,little unsteadily.\n2) Alpha is the threshold level of result set. Defaults to 0.005. \n3) Make sure format and spelling of words are correct.\n4) Results are shown on right side in the below text field.'

top_frame=tkinter.Frame(window,bg='#ffffff')
top_frame.pack(side='top')
input_frame=tkinter.Frame(window,bg='#ffffff')
input_frame.pack(side='left',anchor='nw')
result_frame=tkinter.Frame(window,bg='#ffffff')
result_frame.pack(side='right',anchor='ne')

tkinter.Label(top_frame,text='VSM System K18-1044',font=('Times New Roman',40),bg='dark green',fg='white',width=100).pack()
tkinter.Label(top_frame,text=initial_content,borderwidth=5,relief='sunken',fg='red',font=("Times", "11", "bold"),bg='#ffffff').pack(padx=20,pady=40,ipady=10,ipadx=5)

tkinter.Label(top_frame,text='Input Alpha Default: 0.005',font=('Times New Roman',25,'bold'),bg='#ffffff',fg='green').pack()
alpha_entry=tkinter.Entry(top_frame,textvariable=alpha_input,bg='#ffffff',borderwidth=5,relief='sunken')
alpha_entry.pack(ipadx=200,pady=[4,30],ipady=4)
# query_entry.pack(ipady=5,ipadx=200,padx=50)

tkinter.Label(input_frame,text='Input Query',font=('Times New Roman',25,'bold'),bg='#ffffff',fg='green').pack()
query_entry=tkinter.Entry(input_frame,textvariable=query_input,bg='#ffffff',borderwidth=5,relief='sunken')
query_entry.pack(ipady=5,ipadx=200,padx=50)


tkinter.Label(result_frame,text='Result-set',font=('Times New Roman',25,'bold'),bg='#ffffff',fg='green').pack()
text_area = tkinter.Text(result_frame,height=5,font=('Times New Roman',14),bg='#ffffff',relief='sunken',borderwidth=4)
text_area.pack(side="left",padx=30)
scroll_bar=tkinter.Scrollbar(result_frame,orient="vertical",command=text_area.yview)
scroll_bar.pack(side="left",expand=True, fill="y",padx=10)
text_area.configure(yscrollcommand=scroll_bar.set)


def process_submit_query():
    text_area.delete('1.0',tkinter.END)
    user_query=query_entry.get()
    alpha=alpha_entry.get()
    if alpha=='':
        alpha=0.005
    else:
        alpha=float(alpha)
    if user_query!='' and user_query!='Kindly type something!':
        result=None
        query_vector=model.process_query_vector(user_query)
        result=model.compute_result(query_vector,alpha)
        doc_length=len(result)
        if doc_length==0:
            result='Query resulted in empty set..'
        else:
            result=[ str(num[0]) for num in result ]
            result=','.join(result)

        result+='\nDocuments Received: {}'.format(doc_length)

        text_area.insert(tkinter.END,result)
    else:
        query_input.set('Kindly type something!')    
    

def exit_code():
    quit()


tkinter.Button(input_frame,text='Submit',font=('Times New Roman',12,'bold','underline'),command=process_submit_query,borderwidth=6,bg='#ffffff',fg='green').pack(pady=10,ipadx=15,ipady=5)
tkinter.Button(input_frame,text='Exit',font=('Times New Roman',12,'bold','underline'),command=exit_code,borderwidth=6,bg='#ffffff',fg='red').pack(pady=10,ipadx=15,ipady=5)

window.mainloop()