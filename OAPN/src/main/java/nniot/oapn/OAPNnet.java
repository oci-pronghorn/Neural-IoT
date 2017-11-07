/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nniot.oapn;

import com.ociweb.pronghorn.neural.NeuralGraphBuilder;
import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessFixedFieldPipeConfig;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author nick
 * @author bryson
 */
public class OAPNnet {
    static final int numAttributes = 10;
    static final int numTestRecords = 100;
    static final int numTrainingRecords = 50;
    static final String testDataFN = "";
    static final String trainingDataFN = "";
    static Boolean isTraining = false;
        
    public static void main(String[] args) {
        String [][] trainingData = new String[numTestRecords][numAttributes + 1];
        String [][] testingData =  new String[numTrainingRecords][numAttributes];
        //String []   trainingAnswers = new String[numTrainingRecords];
        
       trainingData= readInData(trainingData, trainingDataFN);
       testingData= readInData(testingData, testDataFN);
        
        GraphManager gm = new GraphManager();
        GraphManager.addDefaultNota(gm, GraphManager.SCHEDULE_RATE, 20_000);
        
        
    }
    
    public static String[][] readInData(String[][] data, String fn){
        String line = null;
        try {
            int i = 0;
            FileReader fileReader = new FileReader(fn);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            while((line = bufferedReader.readLine()) != null) {
                data[i] = line.split(",");
                i++;
            }   
            bufferedReader.close();         
        }
        catch(FileNotFoundException ex) {
            System.out.println("Unable to open file:" + fn);            
        }
        catch(IOException ex) {
            System.out.println("IOException while reading file:" + fn);                
        }
        return data;
    }
    
    //Incomplete, currently based on Nathan's tutorial
    public static void buildVisualNeuralNet(GraphManager gm, String[][] data) {
		final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
		config.hideLabels();
		
		final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();
		
                int inputsCount;
                if(isTraining){
		inputsCount = numAttributes;		
                } else {
                    inputsCount  = numAttributes + 1;
                }
                Pipe<MessageSchemaDynamic>[] prevA = Pipe.buildPipes(inputsCount, config);
		
                //TODO: refer to instance of our stage here
		outputStage.newInstance(gm, data, prevA);
			
		int nodesInLayerA = inputsCount;
		Pipe<MessageSchemaDynamic>[][] fromA = NeuralGraphBuilder.buildPipeLayer(gm, config, prevA, nodesInLayerA, factory);

		int nodesInLayerB = inputsCount;
		Pipe<MessageSchemaDynamic>[][] fromB = NeuralGraphBuilder.buildPipeLayer(gm, config, fromA, nodesInLayerB, factory);
		
		Pipe<MessageSchemaDynamic>[] fromC = NeuralGraphBuilder.lastPipeLayer(gm, fromB, factory);
			
                //TODO: refer to instance of our output stage here
		//DataConsumerStage.newInstance(gm, fromC, target);
		
	}
}
