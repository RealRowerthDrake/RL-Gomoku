from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__, static_folder = "static", template_folder = "templates")
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start',  methods=['POST'])
def start():
    content = request.get_json()
    p1 = content['p1']
    p2 = content['p2']
    return 'success'

@app.route('/getPoint', methods=['GET'])
def getPoint():
    content = request.get_json()

    lastX, lastY = content['lastX'], content['lastY']

from itertools import product
def strategy(board, lastX, lastY, player):
    for i in product(range(board_size), repeat=2):
        if board[i][j] == 0: return (i, j)
    return None

game = GameAPI()

if __name__ == '__main__':
    socketio.run(app)
