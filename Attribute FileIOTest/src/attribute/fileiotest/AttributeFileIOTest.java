/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package attribute.fileiotest;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author nick
 */
public class AttributeFileIOTest {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {
        TestClassWithAttribute[] tArr = new TestClassWithAttribute[10];
        String attrStr, firstRunStr, secRunStr;
        attrStr = "attr.txt";//attributes saved loaded from here
        firstRunStr = "first.txt";//output from when atributes are generated
        secRunStr = "second.txt";//otuput from  when attributes are loaded in
        for (int i = 0; i < tArr.length; i++) {
            tArr[i] = new TestClassWithAttribute();
        }
        Map attrDict = new HashMap<String, Integer>();//key is hash of TestClassWithAttribute
        File fA = new File(attrStr);
        if (fA.exists() && !fA.isDirectory()) {
            BufferedReader b = new BufferedReader(new FileReader(fA));
            String line;
            while ((line = b.readLine()) != null) {
                String k = line.split(" ")[0];
                Integer v = new Integer(line.split(" ")[1]);
                attrDict.put(k, v);

            }
            for (TestClassWithAttribute testClassWithAttribute : tArr) {
                testClassWithAttribute.shouldBeSameAccrossRuns = (int) attrDict.get(testClassWithAttribute.toString());
            }
            try {
                BufferedWriter out = new BufferedWriter(new FileWriter(firstRunStr, false));
                for (TestClassWithAttribute testClassWithAttribute : tArr) {
                    out.write(testClassWithAttribute.toString()+ " "+ testClassWithAttribute.shouldBeSameAccrossRuns+"\n");

                    
                }

                out.close();
            } catch (IOException e) {
            }

        }
        else{
            for (int i = 0; i < tArr.length; i++) {
                tArr[i].shouldBeSameAccrossRuns=i;
                
            }
             try {
                BufferedWriter out = new BufferedWriter(new FileWriter(firstRunStr, false));
                for (TestClassWithAttribute testClassWithAttribute : tArr) {
                    out.write(testClassWithAttribute.toString()+ " "+ testClassWithAttribute.shouldBeSameAccrossRuns+"\n");

                    
                }

                out.close();
            } catch (IOException e) {
            }

            
        }
    }

}
