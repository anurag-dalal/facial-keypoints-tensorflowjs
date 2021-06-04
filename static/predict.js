$("#image-selector").change(function() {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $('#selected-image').attr("src", dataURL);
        $('#prediction-list').empty();
    }
    let file = $('#image-selector').prop('files')[0];
    reader.readAsDataURL(file);
});

let model;
(async function () {
    model = await tf.loadLayersModel('tfjs-model/model.json');
    $('.progress-bar').hide();
})();

$("#predict-button").click(async function () {
    let image = $("#selected-image").get(0);
    //let image = document.querySelector("#selected-image");
    
    let tensor = tf.browser.fromPixels(image)
    .mean(2)
    .toFloat()
    .expandDims(0)
    .expandDims(-1);

    // more pre-processing
    tensor = tf.image.resizeNearestNeighbor(tensor, [96,96]);
    tensor = tensor.div(tf.scalar(255));
    
    let k = await model.predict(tensor).data();
    console.log(k);

    document.getElementById("showkeyp").style.display = 'block';
    document.getElementById("left_eye_center").innerHTML = 'left_eye_center: ('+Math.round(k[0])+','+Math.round(k[1])+')';
    document.getElementById("right_eye_center").innerHTML = 'right_eye_center: ('+Math.round(k[2])+','+Math.round(k[3])+')';
    document.getElementById("left_eye_inner_corner").innerHTML = 'left_eye_inner_corner: ('+Math.round(k[4])+','+Math.round(k[5])+')';
    document.getElementById("left_eye_outer_corner").innerHTML = 'left_eye_outer_corner: ('+Math.round(k[6])+','+Math.round(k[7])+')';
    document.getElementById("right_eye_inner_corner").innerHTML = 'right_eye_inner_corner: ('+Math.round(k[8])+','+Math.round(k[9])+')';
    document.getElementById("right_eye_outer_corner").innerHTML = 'right_eye_outer_corner: ('+Math.round(k[10])+','+Math.round(k[11])+')';
    document.getElementById("left_eyebrow_inner_end").innerHTML = 'left_eyebrow_inner_end: ('+Math.round(k[12])+','+Math.round(k[13])+')';
    document.getElementById("left_eyebrow_outer_end").innerHTML = 'left_eyebrow_outer_end: ('+Math.round(k[14])+','+Math.round(k[15])+')';
    document.getElementById("right_eyebrow_inner_end").innerHTML = 'right_eyebrow_inner_end: ('+Math.round(k[16])+','+Math.round(k[17])+')';
    document.getElementById("right_eyebrow_outer_end").innerHTML = 'right_eyebrow_outer_end: ('+Math.round(k[18])+','+Math.round(k[19])+')';
    document.getElementById("nose_tip").innerHTML = 'nose_tip: ('+Math.round(k[20])+','+Math.round(k[21])+')';
    document.getElementById("mouth_left_corner").innerHTML = 'mouth_left_corner: ('+Math.round(k[22])+','+Math.round(k[23])+')';
    document.getElementById("mouth_right_corner").innerHTML = 'mouth_right_corner: ('+Math.round(k[24])+','+Math.round(k[25])+')';
    document.getElementById("mouth_center_top_lip").innerHTML = 'mouth_center_top_lip: ('+Math.round(k[26])+','+Math.round(k[27])+')';
    document.getElementById("mouth_center_bottom_lip").innerHTML = 'mouth_center_bottom_lip: ('+Math.round(k[28])+','+Math.round(k[29])+')';

    // Create an image element
    var img = new Image();
    var canvas = document.getElementById('c');
    var ctx = canvas.getContext('2d');
    var canover=document.getElementById('cover');

    var ctxover = canvas.getContext('2d');
    img.src = image.src;

    // When the image is loaded, draw it
    img.onload = function () {
        ctx.drawImage(img, 0, 0);
        ctx.fillStyle = 'green';
        var i;
        for (i = 0; i < 15; i++){
            ctxover.fillRect(Math.round(k[i*2]*img.width/96)-4,Math.round(k[i*2+1]*img.height/96)-4,8,8);
        }
    }
    
    canvas.width=img.width;
    canvas.height=img.height;
    canover.width=img.width;
    canover.height=img.height;
})
