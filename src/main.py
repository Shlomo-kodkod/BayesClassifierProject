import logging
from src.menu.menu import Menu

# Configure logging to save all logs to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.FileHandler('app.log', encoding='utf-8')])

if __name__ == "__main__":
    menu = Menu()
    menu.main()
