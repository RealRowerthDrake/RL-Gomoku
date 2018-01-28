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

    elem.appendChild(canvas);
};

Canvas.prototype.start = function () {
    var p1 = $('#1P').val()
    var p2 = $('#2P').val()

    console.log("P1: " + p1 + ', ' + 'P2: ' + p2);

    $.post({
        type: 'POST',
        url: '/start',
        data: JSON.stringify({"p1": p1.toString(), "p2": p2.toString()}),
        contentType: "application/json",
        dataType: 'json'
    });
}

Canvas.prototype.resign = function () {

}

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

    $("#btn-start").click(function() {
        jcanvas.start();
    });

    $("#btn-resign").click(function() {
        jcanvas.resign();
    });
};

