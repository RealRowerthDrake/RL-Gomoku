'use strict';

// Constants

var CHESS = Object.freeze ({
    'CLEAR': 0,
    'BLACK': 1,
    'WHITE': 2
});

function merge_options(obj1,obj2){
    var obj3 = {};
    for (var attrname in obj1) { obj3[attrname] = obj1[attrname]; }
    for (var attrname in obj2) { obj3[attrname] = obj2[attrname]; }
    return obj3;
};

var Board = function(elemId, args) {
    var elem = document.getElementById(elemId);
    var canvas = document.createElement('canvas');

    this.edge = args.padding + args.margin;
    this.grid = args.grid;

    this.width  = canvas.width  = args.grid.x * (args.board_size - 1) + this.edge * 2;
    this.height = canvas.height = args.grid.y * (args.board_size - 1) + this.edge * 2;

    this.radius = args.chess_radius;

    console.log("Width:  " + this.width);
    console.log("Height: " + this.height);

    // draw the background
    this.ctx = canvas.getContext('2d');
    this.ctx.fillStyle = args.board.color;
    this.ctx.fillRect(args.margin, args.margin,
                      this.width - 2 * args.margin, this.height - 2 * args.margin);


    this.ctx.strokeStyle = '#000000';
    this.ctx.lineWidth = 1.2;

    this.ctx.font = 'normal 16px Monaco';
    this.ctx.fillStyle = 'black';
    this.ctx.textAlign = 'center';

    // draw the lines
    for (var i = 0; i < args.board_size; i++){
    	this.ctx.textBaseline = 'bottom';
    	this.ctx.fillText(' ' + (i + 1) + ' ', this.edge + args.grid.x * i, args.margin);

    	this.ctx.beginPath();

    	this.ctx.moveTo(this.edge + args.grid.x * i, this.edge);
    	this.ctx.lineTo(this.edge + args.grid.x * i, this.height - this.edge);

    	this.ctx.stroke();

    	this.ctx.textBaseline = 'top';
    	this.ctx.fillText(' ' + (i + 1) + ' ', this.edge + args.grid.x * i, this.height - args.margin);
    };

    this.ctx.textBaseline = 'middle';
    for (var i = 0; i < args.board_size; i++){
    	this.ctx.textAlign = 'right';
    	this.ctx.fillText(' ' + String.fromCharCode(65 + i) + ' ', args.margin, this.edge + args.grid.y * i);

    	this.ctx.beginPath();

    	this.ctx.moveTo(this.edge,              this.edge + args.grid.y * i);
    	this.ctx.lineTo(this.width - this.edge, this.edge + args.grid.y * i);

    	this.ctx.stroke();

    	this.ctx.textAlign = 'left';
    	this.ctx.fillText(' ' + String.fromCharCode(65 + i) + ' ', this.width - args.margin, this.edge + args.grid.y * i);
    };

    elem.appendChild(canvas);

    this.getCoordinate = function(pageX, pageY) {
        var bounds = canvas.getBoundingClientRect(),
            boardX = (pageX - bounds.left - this.edge),
            boardY = (pageY - bounds.top  - this.edge);

        return {"X": Math.floor( (boardX + 0.5 * args.grid.x) / args.grid.x ),
                "Y": Math.floor( (boardY + 0.5 * args.grid.y) / args.grid.y )}

    }.bind(this);

    canvas.onclick = function(event) {
        if (this.lock || !this.waitForHumanPlayer() || !this.started ){
            return;
        }
        var p = this.getCoordinate(event.clientX, event.clientY);
        this.move(p);
    }.bind(this);
};

Board.prototype.update = function (p, player) {
    this.lock = true;
    $.getJSON( "/update/", {'x': p.X, 'y': p.Y, 'player': player}, function( data ){
        var valid = data.isValid;
        var win   = data.isWin;
    });
    this.lock = false;
    return {'valid': valid, 'win': win};
};

Board.prototype.start = function () {
    this.started = true;
}

Board.prototype.stop = function () {
    this.started = false;
}

Board.prototype.reset = function () {
    this.start();
}

Board.prototype.waitForHumanPlayer = function () {

}

Board.prototype.move = function (p, player) {
    if (this.board[p.X][p.Y] != C.NONE) {
        return;
    }
    ret = this.update(p);
    if (ret.valid) {
        this.board[p.X][p.Y] = player;
        this.drawChess(p, player);
    }
    else {
        window.alert("Invalid Move!");
        return;
    }

    if (ret.win) {
        window.alert(player + " wins!");
        this.stop();
        return;
    }
    else {
        this.step = this.step + 1;
    }
};

Board.prototype.drawChess = function (p, type) {
    ox = p.X * this.grid.x + this.edge;
    oy = p.Y * this.grid.y + this.edge;

    switch (type){
        case CHESS.BLACK:
            this.ctx.fillStyle = '#000000';
            this.ctx.beginPath();
            this.ctx.arc(ox, oy, this.radius, 2*Math.PI, false);
            this.ctx.fill();
            break;

        case CHESS.WHITE:
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.beginPath();
            this.ctx.arc(ox, oy, this.radius, 2*Math.PI, false);
            this.ctx.fill();
            break;
    };
};

// Setup
var Setup = function(elemId, args) {
    console.log("Setup ... ");
    var default_args = {
        chess_radius: 1,
    	board_size: 15,
    	margin:  30,
    	padding: 30,
    	grid:  {x: 35, y: 35},
    	board: {color:'#F7DCB4'}
    };

    self.args = merge_options(default_args, args);
};

Setup.prototype.create = function(elemId) {
    var jboard = new Board(elemId, self.args);
    console.log("Add to " + elemId);
};
