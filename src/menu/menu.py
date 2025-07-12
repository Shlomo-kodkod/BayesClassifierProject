from src.menu.manager import Manager
from src.utils.file_loader import FileLoader
import os


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

    @staticmethod
    def get_choice():
        choice = input()
        return choice

    @staticmethod
    def get_file_path():
        path = input("Enter the csv file path: ")
        return path

    @staticmethod
    def is_file_exist(path):
        return os.path.isfile(FileLoader.build_abs_path(path))

    @staticmethod
    def rum_menu():
        path = Menu.get_file_path()
        if Menu.is_file_exist(path):
            print("Build the model.")
            manager = Manager(path)
            print("testing model accuracy.")
            score = manager.test_model.test_model()
            print(f"Model percent accuracy is: {score}")
        else:
            print("Can't find this file. Try agin later.")
        
    @staticmethod
    def main():
        Menu.display_menu()
        choice = Menu.get_choice()
        if choice == "1":
            Menu.rum_menu()
        print("Exit...")

            

        

   
    

    




