'use strict';

function merge_options(obj1,obj2){
    var obj3 = {};
    for (var attrname in obj1) { obj3[attrname] = obj1[attrname]; }
    for (var attrname in obj2) { obj3[attrname] = obj2[attrname]; }
    return obj3;
}

var Canvas = function(elemId, args) {
    var elem = document.getElementById(elemId);
    var canvas = document.createElement('canvas');

    var edge = args.padding + args.margin;

    this.width  = canvas.width  = args.grid.x * args.board_size + edge * 2;
    this.height = canvas.height = args.grid.y * args.board_size + edge * 2;

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
    for (var i = 0; i <= args.board_size; i++){
    	this.ctx.beginPath();

    	this.ctx.moveTo(edge + args.grid.x * i, edge);
    	this.ctx.lineTo(edge + args.grid.x * i, this.height - edge);

    	this.ctx.stroke();

        if (i < args.board_size){
            this.ctx.textBaseline = 'bottom';
            this.ctx.fillText(' ' + (i + 1) + ' ', edge + args.grid.x * (i + 0.5), args.margin);

            this.ctx.textBaseline = 'top';
            this.ctx.fillText(' ' + (i + 1) + ' ', edge + args.grid.x * (i + 0.5), this.height - args.margin);
        }
    };

    this.ctx.textBaseline = 'middle';
    for (var i = 0; i <= args.board_size; i++){
    	this.ctx.beginPath();

    	this.ctx.moveTo(edge,              edge + args.grid.y * i);
    	this.ctx.lineTo(this.width - edge, edge + args.grid.y * i);

    	this.ctx.stroke();


        if (i < args.board_size){
            this.ctx.textAlign = 'right';
            this.ctx.fillText(' ' + String.fromCharCode(65 + i) + ' ',
                              args.margin,
                              edge + args.grid.y * (i + 0.5));

            this.ctx.textAlign = 'left';
            this.ctx.fillText(' ' + String.fromCharCode(65 + i) + ' ',
                              this.width - args.margin,
                              edge + args.grid.y * (i + 0.5));
        }
    };

    this.backup = this.ctx.getImageData(0, 0, this.width, this.height);

    elem.appendChild(canvas);

    canvas.start = function () {
        var p1 = $('#1P').val()
        var p2 = $('#2P').val()

        this.ctx.putImageData(this.backup, 0, 0);

        $.post({
            type: 'POST',
            url: '/start',
            data: JSON.stringify({"p1": p1, "p2": p2}),
            contentType: "application/json",
            dataType: 'json'
        });
    }.bind(this);

    $("#btn-start").click(function() {
        canvas.start();
    });

    this.getCoordinate = function(pageX, pageY) {
        var bounds = canvas.getBoundingClientRect(),
            boardX = (pageX - bounds.left - edge),
            boardY = (pageY - bounds.top  - edge);

        return {"X": Math.floor( boardX / args.grid.x ),
                "Y": Math.floor( boardY / args.grid.y )}

    }.bind(this);

    var self = this;
    canvas.onclick = function(event) {
        var p = this.getCoordinate(event.clientX, event.clientY);
        var data = {"posX": p.X, "posY": p.Y};

        var ret = $.getJSON("/onClick", data)
            .done (function() {
                var status = ret.responseJSON.status;
                if (status == 'success'){
                    var posX = ret.responseJSON.posX,
                        posY = ret.responseJSON.posY,
                        player = ret.responseJSON.player;

                    canvas.drawChess(posX, posY, player);
                }
            });
    }.bind(this);

    canvas.drawChess = function(posX, posY, player) {
        var ox = edge + (posX + 0.5) * args.grid.x,
            oy = edge + (posY + 0.5) * args.grid.y;

        switch (player) {
            case 0:
                this.ctx.fillStyle = '#000000';
                this.ctx.beginPath();
                this.ctx.arc(ox, oy, args.grid.x / 2, 2*Math.PI, false);
                this.ctx.fill();
                break;

            case 1:
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.beginPath();
                this.ctx.arc(ox, oy, args.grid.x / 2, 2*Math.PI, false);
                this.ctx.fill();
                break;
        }
    }.bind(this);
};

var Setup = function(elemId, args) {
    console.log("Setup ... ");
    var default_args = {
    	board_size: 3,
    	margin:  30,
    	padding: 30,
    	grid:  {x: 35, y: 35},
    	board: {color:'#F7DCB4'}
    };

    self.args = merge_options(default_args, args);
};

Setup.prototype.create = function(elemId) {
    var jcanvas = new Canvas(elemId, self.args);
    console.log("Add to " + elemId);
};

