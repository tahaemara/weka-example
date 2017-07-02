package com.emaraic.ml;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SerializationHelper;

/**
 * This is a classifier for iris.2D.arff dataset  
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: Jul 1, 2017
 */
public class ModelClassifier {

    private Attribute petallength;
    private Attribute petalwidth;

    private ArrayList<Attribute> attributes;
    private ArrayList<String> classVal;
    private Instances dataRaw;


    public ModelClassifier() {
        petallength = new Attribute("petallength");
        petalwidth = new Attribute("petalwidth");
        attributes = new ArrayList<Attribute>();
        classVal = new ArrayList<String>();
        classVal.add("Iris-setosa");
        classVal.add("Iris-versicolor");
        classVal.add("Iris-virginica");

        attributes.add(petallength);
        attributes.add(petalwidth);

        attributes.add(new Attribute("class", classVal));
        dataRaw = new Instances("TestInstances", attributes, 0);
        dataRaw.setClassIndex(dataRaw.numAttributes() - 1);
    }

    /**
     *
     * @param water
     * @param sound
     * @param temp1
     * @param temp2
     * @param result
     * @return Instances
     */
    public Instances createInstance(double petallength, double petalwidth, double result) {
        dataRaw.clear();
        double[] instanceValue1 = new double[]{petallength, petalwidth, 0};
        dataRaw.add(new DenseInstance(1.0, instanceValue1));
        return dataRaw;
    }

    /**
     *
     * @param insts
     * @param path
     * @param type
     * @return String represents class
     */
    public String classifiy(Instances insts, String path) {
        String result = "Not classified!!";
        Classifier cls = null;
        try {
            cls = (MultilayerPerceptron) SerializationHelper.read(path);
            result = classVal.get((int) cls.classifyInstance(insts.firstInstance()));
        } catch (Exception ex) {
            Logger.getLogger(ModelClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }

    /**
     *
     * @return
     */
    public Instances getInstance() {
        return dataRaw;
    }
    
    public static void main(String[] args) {
        ModelClassifier cls=new ModelClassifier();
        System.out.println( cls.classifiy(cls.createInstance(5.1,2, 0),"/Users/Emaraic/Temp/ml/model.data"));
    }

}
