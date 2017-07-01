package com.emaraic.ml;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: Jun 28, 2017
 */
public class ModelGenerator {

    private Instances loadDataset(String path) {
        Instances dataset = null;
        try {
            dataset = DataSource.read(path);
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }

        return dataset;
    }

    public static void main(String[] args) throws Exception {
        ModelGenerator mg = new ModelGenerator();
        Instances dataset = mg.loadDataset("/Users/Emaraic/Temp/ml/iris.2D.arff");

        MultilayerPerceptron m = new MultilayerPerceptron();
        //m.setGUI(true);
        //m.setValidationSetSize(0);
        m.setBatchSize("50");
        m.setLearningRate(0);
        m.setSeed(0);
        m.setMomentum(0);
        m.setTrainingTime(150);//epochs

        /*Multipreceptron parameters and its default values 
        *Learning Rate for the backpropagation algorithm (Value should be between 0 - 1, Default = 0.3).
        *m.setLearningRate(0);
        
	*Momentum Rate for the backpropagation algorithm (Value should be between 0 - 1, Default = 0.2).
	*m.setMomentum(0);
        
        *Number of epochs to train through (Default = 500).
        *m.setTrainingTime(0)
        
	*Percentage size of validation set to use to terminate training (if this is non zero it can pre-empt num of epochs.
	 (Value should be between 0 - 100, Default = 0).
        *m.setValidationSetSize(0);
        
	*The value used to seed the random number generator (Value should be >= 0 and and a long, Default = 0).
        *m.setSeed(0);
        
        *The hidden layers to be created for the network(Value should be a list of comma separated Natural 
	numbers or the letters 'a' = (attribs + classes) / 2, 
	'i' = attribs, 'o' = classes, 't' = attribs .+ classes) for wildcard values, Default = a).
         *m.setHiddenLayers("2,3,3"); three hidden layer with 2 nodes in first layer and 3 nodends in second and 3 nodes in the third.
        
        *The desired batch size for batch prediction  (default 100).
        *m.setBatchSize("1");
         */
        try {
            // divide dataset to train dataset 80% and test dataset 20%
            int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
            int testSize = dataset.numInstances() - trainSize;
            Instances train = new Instances(dataset, 0, trainSize);
            Instances test = new Instances(dataset, trainSize, testSize);
            // build classifier with train dataset             
            m.buildClassifier(train);
           
            // Evaluate classifier with test dataset
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(m, dataset);
            
            System.out.println(eval.toSummaryString("\nResults\n\n", false));
        } catch (Exception ex) {
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
        SerializationHelper.write("/Users/Emaraic/Temp/ml/model.data", m);

    }
}
