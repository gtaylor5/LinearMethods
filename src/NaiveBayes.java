import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class NaiveBayes {

	String dataSetName; //data set name to be trained.
	
	ArrayList<int[]> trainingData = new ArrayList<int[]>(); //usable training data
	ArrayList<int[]> testData = new ArrayList<int[]>(); //usable test data
	ArrayList<Integer> classNumbers = new ArrayList<Integer>(); // class values
	ArrayList<Classification> classifications = new ArrayList<Classification>(); //classifications 
	HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>(); //occurrences of each class
	
	/************************************************************
	Constructor
	************************************************************/
	
	public NaiveBayes(String dataSetName){
		this.dataSetName = dataSetName;
	}
	
	public static void main(String[] args) throws IOException {
		String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
		for(String s : dataSets){
			NaiveBayes b = new NaiveBayes(s);
			System.out.println(s);
			for(int i = 1; i < 6; i++){
				b.fillTrainFile(i);
				b.fillTestFile(i);
				b.countClasses();
				for(int key : b.classCounts.keySet()){
					b.trainNaiveBayes(b.trainingData, key);
				}
				b.testNaiveBayes(b.testData);
			}
			System.out.println();
		}
	}
	
	/************************************************************
	
	Train naive bayes takes in a training set and a class number
	and trains according to the naivebayes algorithm. It first
	sets the probability of the class then sets the probability
	for each attribute in that class. Classifications are added
	to arraylist.
	
	************************************************************/
	
	public void trainNaiveBayes(ArrayList<int[]> trainSet, int classNum){
		Classification _class = new Classification(trainSet, classNum);
		_class.setClassProbability();
		_class.setAttributeProbabilities();
		classifications.add(_class);
		classNumbers.add(classNum);
	}
	
	public void printProbabilities(){
	    for(Classification c : classifications){
	        Main.writer2.println("Probability for Class " + c.classNumber + " : " + c.classProbability);
	        Main.writer2.print("Attribute Probabilities for Class " + c.classNumber + " : ");
	        for(double probs : c.attributeProbabilities){
	            Main.writer2.print(probs + " ");
	        }
	        Main.writer2.println();
	        Main.writer2.println();
	    }
	}
	
	/************************************************************
	
	method tests naive bayes algorithm.
	 * @throws IOException 
	
	************************************************************/
	
	public Double testNaiveBayes(ArrayList<int[]> testSet) throws IOException{
		for(int i = 0; i < classifications.size(); i++){
			for(int j = 0; j < classifications.get(i).attributeProbabilities.length; j++){
			}
		}
		double count = 0;
		double totalCount = 0;
		for(int i = 0; i < testSet.size(); i++){
			double max = Double.MIN_VALUE;
			int maxIndex = 0;
			for(int j = 0; j < classifications.size(); j++){
				double value = classifications.get(j).classProbability;
				for(int k = 0; k < classifications.get(j).attributeProbabilities.length; k++){
					if(testSet.get(i)[k] == 1){
						value*= classifications.get(j).attributeProbabilities[k];
					}else{
						value*= (1.0-classifications.get(j).attributeProbabilities[k]);
					}
				}
				if(value >= max){
					max = value;
					maxIndex = j;
				}
			}
			if(classNumbers.get(maxIndex) == testSet.get(i)[testSet.get(i).length-1]){
				count++;
				totalCount++;
			}else{
				totalCount++;
			}
		}
		return ((double)(count/totalCount)*100);
	}
	
	/************************************************************
	Counts all of the classes in the training set. 
	************************************************************/
	
	public void countClasses(){
		//handle special cases.
		for(int i = 0; i < trainingData.size(); i++){
			if(classCounts.containsKey(trainingData.get(i)[trainingData.get(i).length-1])){ // hashmap contains class?
				int currentVal = classCounts.get(trainingData.get(i)[trainingData.get(i).length-1]); // get current count
				currentVal++; // increment count by 1
				classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], currentVal); //update map.
			}else{
				classCounts.put(trainingData.get(i)[trainingData.get(i).length-1], 1); // add class to map
			}
		}
		//printClassCounts(); // prints class counts.
	}
	
	/************************************************************
	Fills the training file based on the index that is to be skipped.
	Used specifically for cross validation.
	************************************************************/
	
	void fillTrainFile(int indexToSkip) throws IOException{
		for(int i = 1; i < 6; i++){
			if(i == indexToSkip) continue;
			Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+i+".txt"));
			while(fileScanner.hasNextLine()){
				String[] arr = fileScanner.nextLine().split(" ");
				int[] vals = new int[arr.length];
				for(int j = 0; j < arr.length; j++){
					vals[j] = Integer.parseInt(arr[j]);
				}
				trainingData.add(vals);
			}
			fileScanner.close();
		}
	}
	
	/************************************************************
	Fills test file similarly to fillTrainFile.
	************************************************************/
	
	void fillTestFile(int indexToSkip) throws IOException{
		Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+indexToSkip+".txt"));
		while(fileScanner.hasNextLine()){
			String[] arr = fileScanner.nextLine().split(" ");
			int[] vals = new int[arr.length];
			for(int j = 0; j < arr.length; j++){
				vals[j] = Integer.parseInt(arr[j]);
			}
			testData.add(vals);
		}
		fileScanner.close();
	}
}


