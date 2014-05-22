package cat.enric612.weka.test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IB1;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

public class StartWeka {
	
	public static void main(String[] args) throws IOException, Exception {
		
		int i;
		
		/*
		 * Configuracions dels directoris de treball
		 */
		
		// Directori dels models
		String modelsFolder = "Z:/Weka/models/"; 
		
		// Directori de les dades de carrega
		String testFolder = "Z:/Weka/Filtrats/";
		
		/*
		 * Carrega de models
		 */
		
				
		//Model IB1 2n repetició
		Classifier iB12r = (Classifier) weka.core.SerializationHelper.read(modelsFolder+"ib1rep2.model");
		
		//Model IB1 Global resta de dades
		Classifier iB1gr = (Classifier) weka.core.SerializationHelper.read(modelsFolder+"ib1gr.model");
		Classifier mLP10gr = (Classifier) weka.core.SerializationHelper.read(modelsFolder+"mlp10gr.model");
		// Creem un buffer de lectura
		BufferedReader breader = null;
		
		//Instancies d'entrenament
//		breader =  new BufferedReader(new FileReader("Z:/Weka/Filtrats/proves_1_cfsse.arff"));
//		Instances entrenament = new Instances (breader);
//		entrenament.setClassIndex(entrenament.numAttributes() -1);
		
		//Instancies de test
		breader =  new BufferedReader(new FileReader(testFolder+"repeticio2_attributs_9_sensors_no_name.arff"));
		Instances test = new Instances (breader);
		test.setClassIndex(test.numAttributes() -1);
		
		// Tanquem el buffer de lectura
		breader.close();
		
				
		// Cree una copia de les dades de test
		Instances clasificades = new Instances(test);
		
		// Cree una copia de les dades de test
				Instances clasificadesMLP = new Instances(test);
		
		/* 
		 * Test Complet
		 */
		double clsEtiqueta;
		for (i=0;i<test.numInstances();i++){	
			clsEtiqueta = iB1gr.classifyInstance(test.instance(i));
			clasificades.instance(i).setClassValue(clsEtiqueta);
		}
		
		
		
		for (i=0;i<test.numInstances();i++){	
			clsEtiqueta = mLP10gr.classifyInstance(test.instance(i));
			clasificadesMLP.instance(i).setClassValue(clsEtiqueta);
		}
		
		
//		clsEtiqueta = mLPC.classifyInstance(test.instance(1));
//		clasificades.instance(1).setClassValue(clsEtiqueta);
		
		
		
		// Guardem en un nou fitxer de clasificacions
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(testFolder+"resultats_repeticio2_IB1.arff"));
		writer.write(clasificades.toString());
		writer.close();
		
		writer = new BufferedWriter(new FileWriter(testFolder+"resultats_repeticio2_MLP.arff"));
		writer.write(clasificadesMLP.toString());
		writer.close();
		
		
		
		
		
	}

}
