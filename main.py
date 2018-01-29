from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__, static_folder = "static", template_folder = "templates")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start',  methods=['POST'])
def start():
    content = request.get_json()
    p1 = content['p1']
    p2 = content['p2']
    print("#start 1P = {}, 2P = {}".format(p1, p2))

    global board
    board.fill(-1)
    return 'success'

@app.route('/onClick', methods=['GET'])
def onClick():
    print(request.args)
    posX = int(request.args.get("posX"))
    posY = int(request.args.get("posY"))

    global board, turn

    if board[posX][posY] == -1:
        content = {
            "status": "success",
            "posX": posX,
            "posY": posY,
            "player": turn % 2,
        }
        board[posX][posY] = turn % 2
        turn += 1
        return jsonify(content)
    else:
        content = {
            "status": "fail",
        }
        return jsonify(content)

if __name__ == '__main__':
    turn  = 0
    board = np.zeros( (3, 3), dtype = np.int)

    app.run(debug=True)
