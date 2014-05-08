package cat.enric612.weka.test;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IB1;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

public class StartWeka {
	
	public static void main(String[] args) throws IOException, Exception {
		
		// Creem un buffer de lectura
		BufferedReader breader = null;
		
		//Instancies d'entrenament
		breader =  new BufferedReader(new FileReader("Z:/Weka/Filtrats/proves_1_cfsse.arff"));
		Instances entrenament = new Instances (breader);
		entrenament.setClassIndex(entrenament.numAttributes() -1);
		
		//Instancies de test
		breader =  new BufferedReader(new FileReader("Z:/Weka/Filtrats/proves_1_cfsseTest.arff"));
		Instances test = new Instances (breader);
		test.setClassIndex(test.numAttributes() -1);
		
		// Tanquem el buffer de lectura
		breader.close();
		
		// Creem el classificador IB1
		IB1 iB1C = new IB1();
		// Creem el classificador MLP
		MultilayerPerceptron mLPC = new MultilayerPerceptron();
		
		// Entrenem el clasificador amb les dades d'entrenament.
		iB1C.buildClassifier(entrenament);
		mLPC.buildClassifier(entrenament);
		
		// Cree una copia de les dades de test
		Instances clasificades = new Instances(test);
		
		/* Test simple de classificació,
		 * el que farem sera indicar una sola instancia a veure si la classifica correctament 
		 * Repetim el procés per a la mateixa instancia repetida mitjçant un altre classificador.
		 */
		
			
		double clsEtiqueta = iB1C.classifyInstance(test.instance(0));
		clasificades.instance(0).setClassValue(clsEtiqueta);
		clsEtiqueta = mLPC.classifyInstance(test.instance(1));
		clasificades.instance(1).setClassValue(clsEtiqueta);
		
		
		// Guardem en un nou fitxer de clasificacions
		
		BufferedWriter writer = new BufferedWriter(new FileWriter("Z:/Weka/Filtrats/proves_1_cfsseFinal.arff"));
		writer.write(clasificades.toString());
		writer.close();
		System.out.println(clsEtiqueta);
		System.out.println(clasificades.toString());
		
		
		
		
	}

}
