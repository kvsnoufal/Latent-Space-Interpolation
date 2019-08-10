"use strict"
import {MnistData} from './data.js'
console.log("Starting.....")


// getmodel funtion
var encoder;
var decoder;
async function getEncoderModel() {
    encoder  = await await tf.loadLayersModel('/encoder/model.json');
    console.log('Sucessfully loaded encoder');
    // return encoder;
  }
async function getDecoderModel() {
    decoder  = await await tf.loadLayersModel('/decoder/model.json');
    console.log('Sucessfully loaded decoder');
// return decoder;
}

var data;
var buttonImage1=document.getElementById("btn_img1")
var buttonImage2=document.getElementById("btn_img2")
var image1;
var image2;
var promiseimage1;
var promiseimage2;


async function run() {  
  data = new MnistData();
  await data.load();
  // await showExamples(data);
  await getEncoderModel();
  await getDecoderModel();
  
}
run();




buttonImage1.onclick = function(e){console.log("image 1 clicked");
e.preventDefault();
promiseimage1=displayRandomImage1(data,document.getElementById("image1placeholder"))
promiseimage1.then(function(result){image1=result})
}
buttonImage2.onclick = function(e){console.log("image 2 clicked");
e.preventDefault();
promiseimage2=displayRandomImage1(data,document.getElementById("image2placeholder"))
promiseimage2.then(function(result){image2=result})
}


/**
 * Interpolation Button
 * 
 */
var buttonInterp=document.getElementById("btn_interpolate")
buttonInterp.onclick = async function(e){console.log("Interp clicked");
    e.preventDefault();
    try{
        await generate_interpolation(slider.value);
        await generate_actual_image_interpolation(slider.value);}
    catch(err){
        alert("Please generate images first");}
}

var slider=document.getElementById("myRange")
document.getElementById("btn_interpolate_value").value=slider.value

slider.oninput=async function () {
    document.getElementById("btn_interpolate_value").value=this.value
    try{
        await generate_interpolation(this.value);
        await generate_actual_image_interpolation(this.value);}
    catch(err){
        alert("Please generate images first");}
}

async function generate_interpolation(sliderValue) {
    
    let encodedImage1;
    let encodedImage2;
    encodedImage1=await encoder.predict(image1.xs.reshape([1,28,28,1]));
    encodedImage2=await encoder.predict(image2.xs.reshape([1,28,28,1]));
    
    let interpolation=array_interpolation_function(encodedImage1,encodedImage2,sliderValue);
    let decodedInterpolation=decoder.predict(interpolation)
    displayInterpolatedImage(decodedInterpolation,document.getElementById("canvasid"))
    // displayInterpolatedImage(decodedInterpolation,document.getElementById("interpolated_img"))
    

}


async function generate_actual_image_interpolation(sliderValue) {
    
    let actualimage1=await image1.xs.reshape([1,784]);
    let actualimage2=await image2.xs.reshape([1,784]);
    
    let interpolation=array_interpolation_function(actualimage1,actualimage2,sliderValue);
    
    displayActualInterpolatedImage(interpolation,document.getElementById("canvasid"))
    // displayInterpolatedImage(decodedInterpolation,document.getElementById("interpolated_img"))

}


function array_interpolation_function(arr1,arr2,p){
    // console.log("interpolating arrays for " ,p)
    // console.log(arr1)
    let arrayDifference=arr2.sub(arr1);
    let interpolated_array=arr1.add(arrayDifference.mul(p/100));
    return interpolated_array
}
