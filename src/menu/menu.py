from src.menu.manager import Manager
from src.utils.file_loader import FileLoader
import os

# Menu class for user interaction and running the model pipeline
class Menu:
    @staticmethod
    def display_menu():
        print("""
------------------------
   Naive bayes model
------------------------
 [1] Build new model
 [Any key] Exit
 """)

    # Get the user's menu choice
    @staticmethod
    def get_choice():
        choice = input()
        return choice

    # Get the CSV file path
    @staticmethod
    def get_file_path():
        path = input("Enter the csv file path: ")
        return path

    # Get the target column name
    @staticmethod
    def get_target_column():
        target = input("Enter target column: ")
        return target

    # Check if the file exists 
    @staticmethod
    def is_file_exist(path):
        return os.path.isfile(FileLoader.build_abs_path(path))

    # Run the process of loading data, training, and testing the model
    @staticmethod
    def rum_menu():
        path = Menu.get_file_path()
        if Menu.is_file_exist(path):
            target_column = Menu.get_target_column()
            try:
                manager = Manager()
                print("loading data.")
                manager.load_data(path)
                manager.clean_data(manager.data)
                print("Build the model.")
                manager.create_model(target_column)
                print("testing model accuracy.")
                manager.create_model_test()
                score = manager.test_model.test_model()
                print(f"Model percent accuracy is: {score}")
            except Exception as e:
                print(f"Error: {e} column not found.")
        else:
            print("Can't find this file. Try agin later.")
        
    @staticmethod
    def main():
        Menu.display_menu()
        choice = Menu.get_choice()
        if choice == "1":
            Menu.rum_menu()
        print("Exit...")

            

        

   
    

    




