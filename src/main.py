from src.menu.manager import Manager




if __name__ == "__main__":
    manager = Manager()
    score = manager.test_model.test_model()
    print(f"Model percent accuracy is: {score}")