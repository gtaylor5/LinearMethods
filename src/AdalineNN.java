import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class AdalineNN {
    
    AdalineNeuron[] neurons;
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
                AdalineNN a = new AdalineNN();
                a.dataSetName = dataSets[j];
                a.fillTrainFile(i);
                a.fillTestFile(i);
                a.fillValidationSet();
                a.countClasses();
                a.setNeurons();
                a.trainAdaline();
                a.testAdaline();
            }
            System.out.println("-------------------------");
        }
    }
    
    /************************************************************
    Returns performance of perceptron.
    ************************************************************/
    
    public void testAdaline(){
        double performance = 0;
        for(int[] val : testData){
            for(int i = 0; i < neurons.length; i++){
                if(neurons[i].test(val) == 1){
                    performance++;
                }
            }
        }
        System.out.printf("Adaline : %.2f",performance*100/((double)testData.size()));
        System.out.print(" %");
        System.out.println();
    }
    
    /************************************************************
    For each of the neurons in the perceptron train them based on
    their class value.
    ************************************************************/
    
    public void trainAdaline(){
        for(int i = 0; i < neurons.length; i++){
            neurons[i].train();
        }
    }
    
    /************************************************************
    Initialize Array of Neurons with class value and training and
    validation data
    ************************************************************/
    
    public void setNeurons(){
        neurons = new AdalineNeuron[classCounts.size()];
        int i = 0;
        for(Integer key: classCounts.keySet()){
            neurons[i] = new AdalineNeuron(key, trainingData, validationData);
            i++;
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
    Used specifically for cross validation. Also, this method
    inserts an n+1 element equal to -1 while maintaining the class
    label location to be at length-1.
    ************************************************************/
    
    void fillTrainFile(int indexToSkip) throws IOException{
        for(int i = 1; i < 6; i++){
            if(i == indexToSkip) continue; // skip file at index.
            Scanner fileScanner = new Scanner(new File("Data/"+dataSetName+"/Set"+i+".txt"));
            while(fileScanner.hasNextLine()){
                String[] arr = fileScanner.nextLine().split(" ");

                int[] vals = new int[arr.length + 1];
                for(int j = 0; j < vals.length; j++){
                    if(j == arr.length-1){
                        vals[j+1] = Integer.parseInt(arr[j]);
                        vals[j] = -1;
                        break;
                    }
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
            int[] vals = new int[arr.length+1];
            for(int j = 0; j < arr.length; j++){
                if(j == arr.length-1){
                    vals[j+1] = Integer.parseInt(arr[j]);
                    vals[j] = -1;
                    break;
                }
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

            int[] vals = new int[arr.length+1];
            for(int j = 0; j < arr.length; j++){
                if(j == arr.length-1){
                    vals[j+1] = Integer.parseInt(arr[j]);
                    vals[j] = -1;
                    break;
                }
                vals[j] = Integer.parseInt(arr[j]);
            }
            validationData.add(vals);
        }
        fileScanner.close();
    }

}
