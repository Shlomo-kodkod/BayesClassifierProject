import uvicorn
from menu.menu import Menu
from api.app import app
from client.app import App


if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1", port=8000)
