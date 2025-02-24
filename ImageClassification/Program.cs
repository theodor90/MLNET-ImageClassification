using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System.Threading.Channels;
using static Microsoft.ML.DataOperationsCatalog;

public class ImageData
{
    [LoadColumn(0)]
    public string? ImagePath;
    [LoadColumn(1)]
    public string? Label;
}

class InputData
{
    public byte[] Image
    {
        get; set;
    }
    public uint LabelKey
    {
        get; set;
    }
    public string ImagePath
    {
        get; set;
    }
    public string Label
    {
        get; set;
    }
}
class Output
{
    public string ImagePath
    {
        get; set;
    }
    public string Label
    {
        get; set;
    }
    public string PredictedLabel
    {
        get; set;
    }
}


    public class Program
{
    static string dataFolder = "C:\\Users\\theod\\Desktop\\AI Programing in cSHarp\\ImageClassificationApp\\ImageClassification\\Data\\";

    private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder)
    {
        var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);
        foreach (var file in files)
        {
            if ((Path.GetExtension(file) != ".jpg") &&
                (Path.GetExtension(file) != ".png") &&
                (Path.GetExtension(file) != ".jpeg"))
            {
                continue;
            }
            string label = Path.GetFileNameWithoutExtension(file).Trim();
            label = label.Substring(0, label.Length - 1);

            yield return new ImageData()
            {
                ImagePath = file,
                Label = label
            };
        }
    }
    public static void PrintDataView(IDataView dataView)
    {
        var preview = dataView.Preview();
        foreach (var row in preview.RowView)
        {
            foreach (var kvp in row.Values)
            {
                Console.Write($"{kvp.Key}: {kvp.Value} ");
            }
            Console.WriteLine();
        }
        }
    private static void OutputPredictions(Output prediction)
    {
        string imageName = Path.GetFileName(prediction.ImagePath);
        Console.WriteLine($"Image: {imageName} | Actual Label: {prediction.Label} | Predicted Label: {prediction.PredictedLabel}"); 
    }
    private static void ClassifyMultiple(MLContext mlContext, IDataView data, ITransformer trainedModel)
    {
        IDataView predictionData = trainedModel.Transform(data);

        var predictions = mlContext.Data.CreateEnumerable<Output>(predictionData, reuseRowObject: false).ToList();

        Console.WriteLine("AI Predictions: ");
        foreach(var prediction in predictions.Take(4))
        {
            OutputPredictions(prediction);
        }
    }
    public static void Main()
    {
        MLContext mlContext = new MLContext();
        IEnumerable<ImageData> images = LoadImagesFromDirectory(dataFolder);
        IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

        IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

        //PrintDataView(shuffledData);

        var preprocessingPipeline = mlContext.Transforms.Conversion
            .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
            .Append(mlContext.Transforms.LoadRawImageBytes("Image", dataFolder, "ImagePath"));

        IDataView preprocessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

        TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.4);
        IDataView trainSet = trainTestSplit.TrainSet;
        IDataView testSet = trainTestSplit.TestSet;

        var classifierOptions = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelKey",
            Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
            MetricsCallback = Console.WriteLine,
            TestOnTrainSet = false,
            ValidationSet = testSet,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true,
        };
        var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        ITransformer trainedModel = trainingPipeline.Fit(trainSet);

        ClassifyMultiple(mlContext, testSet, trainedModel);
    }
}
