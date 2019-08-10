

async function displayRandomImage1(data,sf) {
    let surface=sf;
    sf.innerHTML='';

    const randimage = data.nextTestBatch(1)
    let tempImageStorage=randimage
    const imageTensor = tf.tidy(() => {
        // Reshape the image to 28x28 px
        return randimage.xs
          .slice([0, 0], [1, randimage.xs.shape[1]])
          .reshape([28, 28, 1]);
      });
    const canvas = document.createElement('canvas');
    canvas.width = 100;
    canvas.height = 100;
    canvas.style = 'margin: 4px;';
    const imageTensor_res=await tf.image.resizeBilinear(imageTensor,[200,200])
    await tf.browser.toPixels(imageTensor_res, canvas);
    surface.appendChild(canvas);

    imageTensor.dispose();
    // console.log(randimage)
    return randimage
}

async function displayInterpolatedImage(randimage,sf) {
    let surface=sf;
    
    const imageTensor = tf.tidy(() => {
        // Reshape the image to 28x28 px
        return randimage
          .slice([0, 0], [1, randimage.shape[1]])
          .reshape([28, 28, 1]);
      });
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    canvas.style = 'margin: 4px;';
    canvas.setAttribute("id", "canvasid");
    // clearImage()
    // console.log(imageTensor)
    const imageTensor_res=await tf.image.resizeBilinear(imageTensor,[224,224])
    await tf.browser.toPixels(imageTensor_res, canvas);
    document.getElementById("interpolated_img").innerHTML= await '';
    await document.getElementById("interpolated_img").appendChild(canvas);

    await imageTensor.dispose();

}
async function displayActualInterpolatedImage(randimage,sf) {
    let surface=sf;
    
    const imageTensor = tf.tidy(() => {
        // Reshape the image to 28x28 px
        return randimage
          .slice([0, 0], [1, randimage.shape[1]])
          .reshape([28, 28, 1]);
      });
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    canvas.style = 'margin: 4px;';
    canvas.setAttribute("id", "canvasid");
    // clearImage()
    // console.log(imageTensor)
    const imageTensor_res=await tf.image.resizeBilinear(imageTensor,[224,224])
    await tf.browser.toPixels(imageTensor_res, canvas);
    document.getElementById("actualinterpolated_img").innerHTML= await '';
    await document.getElementById("actualinterpolated_img").appendChild(canvas);

    await imageTensor.dispose();

}