import "./styles.css";
require("babel-core/register");
require("babel-polyfill");
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
// import * as Papa from "papaparse";
// import * as cars from "./data/carsData";
import * as Plotly from "plotly.js-dist";

async function getData() {
  const carsDataReq = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  // const carsData = cars.default;
  const carsData = await carsDataReq.json();
  const cleaned = carsData
    .map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
      wieght: car.Weight_in_lbs,
      acceleration: car.Acceleration,
      displacement: car.Displacement
    }))
    .filter(
      car =>
        car.mpg != null &&
        car.horsepower != null &&
        car.wieght != null &&
        car.acceleration != null &&
        car.displacement != null
    );

 
  renderScatter("qual-cont", carsData, ["Horsepower", "Miles_per_Gallon"], {
    title: "Miles/Gallon vs HorsePower",
    xLabel: "Horsepower",
    yLabel: "Miles per Gallon"
  });
  renderScatter(
    "liv-area-cont",
    carsData,
    ["Weight_in_lbs", "Miles_per_Gallon"],
    {
      title: "Miles/Gallon vs Weight_in_lbs",
      xLabel: "Weight_in_lbs",
      yLabel: "Miles per Gallon"
    }
  );
  renderScatter("year-cont", carsData, ["Acceleration", "Miles_per_Gallon"], {
    title: "Miles/Gallon vs Acceleration",
    xLabel: "Acceleration",
    yLabel: "Miles per Gallon"
  });
  renderScatter(
    "year-price-cont",
    carsData,
    ["Displacement", "Miles_per_Gallon"],
    {
      title: "Miles/Gallon vs Displacement",
      xLabel: "Displacement",
      yLabel: "Miles per Gallon"
    }
  );

  return cleaned;
}
const features = ["displacement", "acceleration", "wieght", "horsepower"];
async function runFirst() {
  const data = await getData();

  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  await trainModel(model, inputs, labels);
  console.log("Done Training");

  testModel(model, data, tensorData);
}
function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [4], units: 50 }));
  model.add(tf.layers.dense({ units: 100, activation: "sigmoid" }));
  model.add(
    tf.layers.dense({ units: 1, useBias: true, activation: "sigmoid" })
  );

  return model;
}

const arrDis = [];
const arrAcc = [];
const arrWei = [];
const arrHor = [];
function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    data.map(d => {
      arrDis.push(d["displacement"]);
      arrAcc.push(d["acceleration"]);
      arrWei.push(d["wieght"]);
      arrHor.push(d["horsepower"]);
    });
    const minDis = Math.min(...arrDis);
    const maxDis = Math.max(...arrDis);
    const minAcc = Math.min(...arrAcc);
    const maxAcc = Math.max(...arrAcc);
    const minWei = Math.min(...arrWei);
    const maxWei = Math.max(...arrWei);
    const minHor = Math.min(...arrHor);
    const maxHor = Math.max(...arrHor);

    data.map(d => {
      d["displacement"] = (d["displacement"] - minDis) / (maxDis - minDis);
      d["acceleration"] = (d["acceleration"] - minAcc) / (maxAcc - minAcc);
      d["wieght"] = (d["wieght"] - minWei) / (maxWei - minWei);
      d["horsepower"] = (d["horsepower"] - minHor) / (maxHor - minHor);
    });
    const inputs = data.flatMap(r =>
      features.flatMap(f => {
        //   if (categoricalFeatures.has(f)) {
        //     return oneHot(!r[f] ? 0 : r[f], VARIABLE_CATEGORY_COUNT[f]);
        //   }
        return !r[f] ? 0 : r[f];
      })
    );
    let labels = data.map(d => d.mpg);
    const inputTensor = tf.tensor2d(inputs, [inputs.length / 4, 4]);
    const labelTensor = tf.tensor(labels);

    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: inputTensor,
      labels: normalizedLabels,
      labelMax,
      labelMin,
      minDis,
      maxDis,
      minAcc,
      maxAcc,
      minWei,
      maxWei,
      minHor,
      maxHor
    };
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"]
  });

  const batchSize = 32;
  const epochs = 30;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    )
  });
}
function testModel(model, inputData, normalizationData) {
  const { labelMin, labelMax } = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const da = normalizationData.inputs
    .dataSync()
    .filter(function(value, index, Arr) {
      return index % 4 == 0;
    });
  const [preds] = tf.tidy(() => {
    // const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(normalizationData.inputs);
    // const maxTensor = tf.scalar(normalizationData.maxDis);
    // const minTensor = tf.scalar(normalizationData.minDis);
    // const daTensor = tf.tensor(da);
    // const unNormXs = daTensor.mul(maxTensor.sub(minTensor)).add(minTensor);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    // Un-normalize the data
    return [unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(arrHor).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d, i) => ({
    x: arrHor[i],
    y: d.mpg
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"]
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300
    }
  );
}
document.addEventListener("DOMContentLoaded", runFirst);

const renderHistogram = (container, data, column, config) => {
  const columnData = data.map(r => r[column]);

  const columnTrace = {
    name: column,
    x: columnData,
    type: "histogram",
    opacity: 0.7,
    marker: {
      color: "dodgerblue"
    }
  };

  Plotly.newPlot(container, [columnTrace], {
    xaxis: {
      title: config.xLabel,
      range: config.range
    },
    yaxis: { title: "Count" },
    title: config.title
  });
};

const renderScatter = (container, data, columns, config) => {
  var trace = {
    x: data.map(r => r[columns[0]]),
    y: data.map(r => r[columns[1]]),
    mode: "markers",
    type: "scatter",
    opacity: 0.7,
    marker: {
      color: "dodgerblue"
    }
  };

  var chartData = [trace];

  Plotly.newPlot(container, chartData, {
    title: config.title,
    dragmode: 'select',
    hovermode:'closest',
    xaxis: {
      title: config.xLabel
    },
    yaxis: { title: config.yLabel }
  });

  let myPlot = document.getElementById(container);
  myPlot.on('plotly_selected', function(selecteddata) {
    const x = [];
    const y = [];
    alert("You clicked this Plotly chart!");
    selecteddata.points.forEach(data => {
      x.push(data.x);
      y.push(data.y);
    });
    document.getElementById('qual-data').innerText = mean(x);
    console.log(data);
  });
};

function mean(numbers) {
  var total = 0, i;
  for (i = 0; i < numbers.length; i += 1) {
      total += numbers[i];
  }
  return total / numbers.length;
}

const renderPredictions = (trueValues, slmPredictions, lmPredictions) => {
  var trace = {
    x: [...Array(trueValues.length).keys()],
    y: trueValues,
    mode: "lines+markers",
    type: "scatter",
    name: "true",
    opacity: 0.5,
    marker: {
      color: "dodgerblue"
    }
  };

  var slmTrace = {
    x: [...Array(trueValues.length).keys()],
    y: slmPredictions,
    name: "pred",
    mode: "lines+markers",
    type: "scatter",
    opacity: 0.5,
    marker: {
      color: "forestgreen"
    }
  };

  var lmTrace = {
    x: [...Array(trueValues.length).keys()],
    y: lmPredictions,
    name: "pred",
    mode: "lines+markers",
    type: "scatter",
    opacity: 0.5,
    marker: {
      color: "forestgreen"
    }
  };

  Plotly.newPlot("slm-predictions-cont", [trace, slmTrace], {
    title: "Simple Linear Regression predictions",
    yaxis: { title: "Price" }
  });

  Plotly.newPlot("lm-predictions-cont", [trace, lmTrace], {
    title: "Linear Regression predictions",
    yaxis: { title: "Price" }
  });
};

const VARIABLE_CATEGORY_COUNT = {
  OverallQual: 10,
  GarageCars: 5,
  FullBath: 4
};
// -----------------------------------------------------------------------------
// Papa.parsePromise = function(file) {
//   return new Promise(function(complete, error) {
//     Papa.parse(file, {
//       header: true,
//       download: true,
//       dynamicTyping: true,
//       complete,
//       error
//     });
//   });
// };

// const prepareData = async () => {
//   const csv = await Papa.parsePromise(
//     "https://docs.google.com/spreadsheets/d/1q5nNJ0kUD9sPdG71YB0MFopzzE9iRkQ7j6I_KLbfPW8/export?format=csv"

//   );
//   return csv.data;
// };

// // normalized = (value − min_value) / (max_value − min_value)
// const normalize = tensor =>
//   tf.div(
//     tf.sub(tensor, tf.min(tensor)),
//     tf.sub(tf.max(tensor), tf.min(tensor))
//   );

// const oneHot = (val, categoryCount) =>
//   Array.from(tf.oneHot(val, categoryCount).dataSync());

// const createDataSets = (data, features, categoricalFeatures, testSize) => {
//   const X = data.map(r =>
//     features.flatMap(f => {
//       if (categoricalFeatures.has(f)) {
//         return oneHot(!r[f] ? 0 : r[f], VARIABLE_CATEGORY_COUNT[f]);
//       }
//       return !r[f] ? 0 : r[f];
//     })
//   );

//   const X_t = normalize(tf.tensor2d(X));

//   const y = tf.tensor(data.map(r => (!r.SalePrice ? 0 : r.SalePrice)));

//   const splitIdx = parseInt((1 - testSize) * data.length, 10);

//   const [xTrain, xTest] = tf.split(X_t, [splitIdx, data.length - splitIdx]);
//   const [yTrain, yTest] = tf.split(y, [splitIdx, data.length - splitIdx]);

//   return [xTrain, xTest, yTrain, yTest];
// };

// const trainLinearModel = async (xTrain, yTrain) => {
//   const model = tf.sequential();

//   model.add(
//     tf.layers.dense({
//       inputShape: [xTrain.shape[1]],
//       units: xTrain.shape[1],
//       activation: "sigmoid"
//     })
//   );
//   model.add(tf.layers.dense({ units: 1 }));

//   model.compile({
//     optimizer: tf.train.sgd(0.001),
//     loss: "meanSquaredError",
//     metrics: [tf.metrics.meanAbsoluteError]
//   });

//   const trainLogs = [];
//   const lossContainer = document.getElementById("loss-cont");
//   const accContainer = document.getElementById("acc-cont");

//   await model.fit(xTrain, yTrain, {
//     batchSize: 32,
//     epochs: 30,
//     shuffle: true,
//     validationSplit: 0.1,
//     callbacks: {
//       onEpochEnd: async (epoch, logs) => {
//         trainLogs.push({
//           rmse: Math.sqrt(logs.loss),
//           val_rmse: Math.sqrt(logs.val_loss),
//           mae: logs.meanAbsoluteError,
//           val_mae: logs.val_meanAbsoluteError
//         });
//         tfvis.show.history(lossContainer, trainLogs, ["rmse", "val_rmse"]);
//         tfvis.show.history(accContainer, trainLogs, ["mae", "val_mae"]);
//       }
//     }
//   });

//   return model;
// };

// const run = async () => {
//   const data = await prepareData();

//   renderHistogram("qual-cont", data, "OverallQual", {
//     title: "Overall material and finish quality (0-10)",
//     xLabel: "Score"
//   });

//   renderHistogram("liv-area-cont", data, "GrLivArea", {
//     title: "Above grade (ground) living area square feet",
//     xLabel: "Area (sq. ft)"
//   });

//   renderHistogram("year-cont", data, "YearBuilt", {
//     title: "Original construction date",
//     xLabel: "Year"
//   });

//   renderScatter("year-price-cont", data, ["YearBuilt", "SalePrice"], {
//     title: "Year Built vs Price",
//     xLabel: "Year",
//     yLabel: "Price"
//   });

//   renderScatter("qual-price-cont", data, ["OverallQual", "SalePrice"], {
//     title: "Quality vs Price",
//     xLabel: "Quality",
//     yLabel: "Price"
//   });

//   renderScatter("livarea-price-cont", data, ["GrLivArea", "SalePrice"], {
//     title: "Living Area vs Price",
//     xLabel: "Living Area",
//     yLabel: "Price"
//   });

//   const [
//     xTrainSimple,
//     xTestSimple,
//     yTrainSimple,
//     yTestIgnored
//   ] = createDataSets(data, ["GrLivArea"], new Set(), 0.1);
//   const simpleLinearModel = await trainLinearModel(xTrainSimple, yTrainSimple);

//   const features = [
//     "OverallQual",
//     "GrLivArea",
//     "GarageCars",
//     "TotalBsmtSF",
//     "FullBath",
//     "YearBuilt"
//   ];
//   const categoricalFeatures = new Set([
//     "OverallQual",
//     "GarageCars",
//     "FullBath"
//   ]);
//   const [xTrain, xTest, yTrain, yTest] = createDataSets(
//     data,
//     features,
//     categoricalFeatures,
//     0.1
//   );
//   console.log(xTrain);
//   const linearModel = await trainLinearModel(xTrain, yTrain);

//   const trueValues = yTest.dataSync();
//   const slmPreds = simpleLinearModel.predict(xTestSimple).dataSync();
//   const lmPreds = linearModel.predict(xTest).dataSync();

//   renderPredictions(trueValues, slmPreds, lmPreds);
// };

// if (document.readyState !== "loading") {
//   run();
// } else {
//   document.addEventListener("DOMContentLoaded", run);
// }
