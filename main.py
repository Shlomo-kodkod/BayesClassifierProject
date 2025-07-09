import pandas as pd
from tkinter import Tk, filedialog
import ctypes

if __name__ == "__main__":


    tk = Tk()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  
    except:
        pass
    tk.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="chose CSV file",
        filetypes=[("CSV Files", "*.csv")]
    )
    # df = pd.read_csv(r'data\buy_computer_data.csv')
    if file_path:
        try:
            df = pd.read_csv(file_path)
            print("success")
        except Exception as e:
            print(f"failed: {e}")
    else:
        print("file not uploaded.")

    data_columns = df.columns[:-1]  
    target_column = df.columns[-1]  
    len_target_col = df.iloc[:,-1].count()
    values_map = (df[target_column].value_counts() / df[target_column].count()).to_dict()
    model_data = {}

    for cls in values_map:
        df_cls = df[df[target_column] == cls] 
        model_data[cls] = dict()

        for feature in data_columns:
            model_data[cls][feature] = ((df_cls[feature].value_counts() + 1) / (df_cls[feature].count() + len(df_cls[feature].unique()))).to_dict()

    print(model_data)

    

