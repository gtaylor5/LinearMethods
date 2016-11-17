import java.io.File;
import java.io.IOException;
import java.util.*;
public class Perceptron {

    Neuron[] neurons;
    ArrayList<int[]> trainingData = new ArrayList<int[]>();
    ArrayList<int[]> testData = new ArrayList<int[]>();
    ArrayList<int[]> validationData = new ArrayList<int[]>();
    HashMap<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
    String dataSetName = "";
    
    
    
    public static void main(String[] args) throws IOException {
        String[] dataSets = {"SoyBean","Iris","GlassID","BreastCancer","VoteCount"};
        for(int j = 0; j < 5; j++){
            System.out.println(dataSets[j] + ": ");
            for(int i = 1; i < 6; i++){
                Perceptron p = new Perceptron();
                p.dataSetName = dataSets[j];
                p.fillTrainFile(i);
                p.fillTestFile(i);
                p.fillValidationSet();
                p.countClasses();
                p.setNeurons();
                p.trainPerceptron();
                p.testPerceptron();
            }
            System.out.println("-------------------------");
        }
    }
    
    /************************************************************
    Returns performance of perceptron.
    ************************************************************/
    
    public double testPerceptron(){
        double performance = 0;
        for(int[] val : testData){
            for(int i = 0; i < neurons.length; i++){
                if(neurons[i].test(val) == 1){
                    performance++;
                }
            }
        }
        return performance*100/((double)testData.size());
    }
    
    /************************************************************
    For each of the neurons in the perceptron train them based on
    their class value.
    ************************************************************/
    
    public void trainPerceptron(){
        for(int i = 0; i < neurons.length; i++){
            neurons[i].train();
        }
    }
    
    /************************************************************
    Initialize Array of Neurons with class value and training and
    validation data
    ************************************************************/
    
    public void setNeurons(){
        neurons = new Neuron[classCounts.size()];
        int i = 0;
        for(Integer key: classCounts.keySet()){
            neurons[i] = new Neuron(key, trainingData, validationData);
            i++;
        }
    }
    
    public void printWeights(){
        for(Neuron n : neurons){
            Main.writer2.print("Weights for Class " + n.classVal + " : ");
            for(double w : n.weights){
                Main.writer2.print(w + " ");
            }
            Main.writer2.println();
            Main.writer2.println();
        }
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
    }
    
    /************************************************************
    Fills the training file based on the index that is to be skipped.
    Used specifically for cross validation.
    ************************************************************/
    
    void fillTrainFile(int indexToSkip) throws IOException{
        for(int i = 1; i < 6; i++){
            if(i == indexToSkip) continue; // skip file at index.
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
    
    /************************************************************
    Fills the validation set.
    ************************************************************/
    void fillValidationSet() throws IOException{
        Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/validationSet.txt"));
        while(fileScanner.hasNextLine()){
            String[] arr = fileScanner.nextLine().split(" ");

            int[] vals = new int[arr.length];
            for(int j = 0; j < arr.length; j++){
                vals[j] = Integer.parseInt(arr[j]);
            }
            validationData.add(vals);
        }
        fileScanner.close();
    }
    
}
