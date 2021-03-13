var net;
var count = 0;
const imgE1 = document.getElementById("img");
const descE1 = document.getElementById("descripcion-imagen");
const Probabilidad = document.getElementById("probabilidad");
//const file = document.getElementById("uploadF");
var archivo = document.getElementById("file");

const webcamE1 = document.getElementById("webcam1");
const descCam1 = document.getElementById("console");

const clasificador = knnClassifier.create();
var webcam;

async function app() {
    net = await mobilenet.load();
    const result = await net.classify(imgE1);
    console.log(result);
       ///AGREGANDO LA CAMARA
    descE1.innerHTML = JSON.stringify(result);
    webcam = await tf.data.webcam(webcamE1);

  while(true){
    const img = await webcam.capture();
    const result = await net.classify(img);
    const activation = net.infer(img,"conv_preds");
    var result2;

    try{
      result2 = await clasificador.predictClass(activation);
    }catch(error){
      result2 = {}
    }

    const clases = ["No entrenado","Sankayo","P’enqa-p’enqa","Cantuta rosada","Pilli, Misk’ipilli (quechua), Harp’a (aymara)"];

    descCam1.innerHTML = "<h4>PREDICCION</h4><p>" + result[0].className +
    "</p><h4>PROBABILIDAD</h4><p>" + result[0].probability + "</p>";

    /*
    if(PalAnterior != clases[result2.label]&&PalAnterior2 != clases[result2.label]){

      palabras = palabras+" "+clases[result2.label];
      PalAnterior2 = PalAnterior;
      PalAnterior = clases[result2.label];

    }

    try{

      document.getElementById("console2").innerHTML="<h4>PREDICCION DEL CLASIFICADOR</h4><h5 style='color: #00B92F;'>" + clases[result2.label] + "</h5>";
      
      document.getElementById("textoPalabras").innerHTML="<p class='text-break' style='width: 600px;margin-top: 10px;font-style: oblique;font-size: 32px;color: #0089C2;'>" + palabras + "</p>";
      
      
    }catch(error){
      document.getElementById("console2").innerHTML="No entrenado";
    }*/
    try {
        document.getElementById("console2").innerHTML="preddiciohn del clasificador:" + clases[result2.label];
    } catch (error) {
        document.getElementById("console2").innerHTML="no entrenado";
    }

    img.dispose();
    await tf.nextFrame(); 
  }

}

imgE1.onload = async function() {
    displayImagePrediction();
}

async function displayImagePrediction() {
    try {
        const result = await net.classify(imgE1);
        //descE1.innerHTML = JSON.stringify(result);
        descE1.innerText = result[0].className;
        Probabilidad.innerText = result[0].probability;
        
    } catch (error) {

    }
}


async function cambiarImagen() {
    count = count + 1;
    imgE1.src = "https://picsum.photos/200/300?random=" + count;
    descE1.innerHTML = "";
    console.log(descE1);
}

async function addExample(classId){
    console.log("Ejemplo agregado");
    const img = await webcam.capture();
    const activation = net.infer(img,true);
    clasificador.addExample(activation, classId);
    //await model.save('localstorage://my-model');
    img.dispose();
  }
  
async function cargarImagen(){
  
}
/*async function UploadF(){
    var reader = new FileReader();
    if (file){
        reader.readAsDataURL(archivo);
        reader.onloadend = function (){
            document.getElementById("img").src =reader.result
            imgE1.i
        }
    }


}*/

app();