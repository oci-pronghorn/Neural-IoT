package nniot.oapn;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import static com.ociweb.pronghorn.stage.PronghornStage.NONE;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class OutputStage extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] input;
    private final Float[][] data;
    private float result;

    public OutputStage(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, NONE);
        this.input = new Pipe[]{input};
        this.data = null;
    }
    
    public OutputStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, output);
        this.input = input;
        this.data = null;
    }

    public OutputStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
        super(gm, input, output);
        this.input = input;
        this.data = null;
    }

    @Override
    public void run() {
//        try {
//            writeOutput();
//        } catch (FileNotFoundException ex) {
//            Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
//        } catch (IOException ex) {
//            Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
//        }

    }
}
//
//    /* public int writeOutput() {
//     int result = Integer.MAX_VALUE;
//     int i = output.length;
//     while (--i>=0) {
//     result = Math.min(result, SchemalessPipe.roomRemaining(output[i]));
//     }
//     return result;
//     }*/
//    public void writeOutput() throws FileNotFoundException, IOException {
//        for (Float[] row : data) {
//            String s = "";
//            for (Float item : row) {
//                s += item.toString();
//                s += ", ";
//            }
//            s += getCorrelatedOutput(row);
//            s += "\n";
//            outputFileWriter.write(s);
//        }
//        outputFileWriter.close();
//        if (OAPNnet.isTraining) {
//            BufferedWriter out = new BufferedWriter(new FileWriter(weightsFile, false));
//            for (int i = 0; i < OAPNnet.nodesByLayer.size(); i++) {
//                for (int j = 0; j < OAPNnet.nodesByLayer.get(i).length; j++) {
//                    VisualNode node = OAPNnet.nodesByLayer.get(i)[j];
//                    String pipeWeight;
//                    out.write(node.stageId);
//                    for (int k = 0; k < node.input.length; k++) {
//                        out.write(" " + node.input[k].toString() + "," + node.getWeight(k) + " ");
//                    }
//                    out.write("\n");
//                }
//            }
//
//            out.close();
//            out = new BufferedWriter(new FileWriter(biasesFile, false));
//            for (int i = 0; i < OAPNnet.nodesByLayer.size(); i++) {
//                for (int j = 0; j < OAPNnet.nodesByLayer.get(i).length; j++) {
//                    VisualNode node = OAPNnet.nodesByLayer.get(i)[j];
//                    out.write(node.stageId + " " + node.getBias() + "\n");
//                }
//            }
//
//            out.close();
//
//        }
//    }
//
//    private String getCorrelatedOutput(Float s[]) {
//        // ask dr mayer and mr tippy
//        //TODO: WHICH INPUT ROW GOES WITH WHICH OUTOUT, HASH EACH INPUT LINBE AND USE SINGLETON HASHMAP
//        return null;
//    }
