package com.example.myapplication;

import android.app.AlertDialog;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import android.widget.Toast;

import androidx.core.app.NotificationCompat;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.gpu.GpuDelegate;


import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Objects;


public class Inference extends Service {
    private AlertDialog alertDialog;

    public Inference() {
    }

    private Interpreter tflite;

    public static String[] getAssetFiles(Context context, String folderPath) {
        AssetManager assetManager = context.getAssets();
        String[] files = null;
        try {
            files = assetManager.list(folderPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return files;
    }

    public static final String CHANNEL_ID = "ForegroundServiceChannel";

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel serviceChannel = new NotificationChannel(
                    CHANNEL_ID,
                    "Foreground Service Channel",
                    NotificationManager.IMPORTANCE_DEFAULT
            );
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(serviceChannel);
        }
    }


    public void TFLiteInference(int tid) throws IOException, InterruptedException {
        CompatibilityList compatList = new CompatibilityList();
        Interpreter.Options gpu_options = new Interpreter.Options();
        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate gpuDelegate = new GpuDelegate();
            gpu_options.addDelegate(gpuDelegate);
        }

        String[] models = getAssetFiles(Inference.this, "");

        // Define the dimensions of the input tensor
        int inputWidth = 96;
        int inputHeight = 96;
        int inputChannels = 1;

        float[][][][] inputArray = new float[1][inputHeight][inputWidth][inputChannels];
        for (int i = 0; i < inputHeight; i++) {
            for (int j = 0; j < inputWidth; j++) {
                for (int k = 0; k < inputChannels; k++) {
                    inputArray[0][i][j][k] = (float) Math.random();
                }
            }
        }

        int outputSize = 50; // Assuming the output size is 5
        float[][] outputArray = new float[1][outputSize];
        String TAG = "tfliteinference";

//        Interpreter tflite_cpu= new Interpreter(FileUtil.loadMappedFile(Inference.this,"cnn.tflite"));
//
//        Interpreter tflite_gpu = new Interpreter(FileUtil.loadMappedFile(Inference.this,"cnn.tflite"),gpu_options);

//        long[] cpu_sum = new long [51];
//        long[] gpu_sum = new long [51];

//        int time = 0;
//        while(time++ < 5000){
//
//            for (int i = 0; i < inputHeight; i++) {
//                for (int j = 0; j < inputWidth; j++) {
//                    for (int k = 0; k < inputChannels; k++) {
//                        inputArray[0][i][j][k] = (float) Math.random();
//                    }
//                }
//            }
//
//
//            long t3 = System.currentTimeMillis();
//            tflite_cpu.run(inputArray, outputArray);
//            long t4 = System.currentTimeMillis();
//            Log.d(TAG, tid + "cpu takes " +  (t4-t3)  + "ms");
//
//            t3 = System.currentTimeMillis();
//            tflite_gpu.run(inputArray, outputArray);
//            t4 = System.currentTimeMillis();
//            Log.d(TAG, tid + "gpu takes " +  (t4-t3)  + "ms");
//
//            t3 = System.currentTimeMillis();
//            tflite_cpu.run(inputArray, outputArray);
//            t4 = System.currentTimeMillis();
//            Log.d(TAG, tid + "cpu takes " +  (t4-t3)  + "ms");
//            cpu_sum[time] = t4-t3 ;
//
//            long t5 = System.currentTimeMillis();
//            tflite_gpu.run(inputArray, outputArray);
//            long t6 = System.currentTimeMillis();
//            Log.d(TAG, tid + "gpu takes " +  (t4-t3)  + "ms");
//            gpu_sum[time] = t4-t3 ;
//
//            Log.i(TAG, (t4-t3) + "," + (t6-t5));
//        }
//
//        int cpu_total = 0;
//        int gpu_total = 0;
//
//        for(int i =0 ;i < 50; i++){
//            cpu_total += cpu_sum[i];
//            gpu_total += gpu_sum[i];
//        }
//        Log.d(TAG, Float.toString((float) (cpu_total / 50.0)) + "," + Float.toString((float) (gpu_total/ 50.0)));
//
//        tflite_cpu.close();
//        tflite_gpu.close();

//
//        CommandExecution.easyExec("taskset -p f0 " + tid,true);
//        String output = CommandExecution.execCommand("taskset -p " + tid,true).successMsg;
//        Log.d(TAG, output);
//

//        long t3 = System.currentTimeMillis();
//        tflite_cpu.run(inputArray, outputArray);
//        long t4 = System.currentTimeMillis();
//        Log.d(TAG, tid + "cpu  takes " +  (t4-t3)  + "ms");
//
//        t3 = System.currentTimeMillis();
//        tflite_gpu.run(inputArray, outputArray);
//        t4 = System.currentTimeMillis();
//        Log.d(TAG, tid + "gpu takes " +  (t4-t3)  + "ms");
//
//        t3 = System.currentTimeMillis();
//        tflite_cpu.run(inputArray, outputArray);
//        t4 = System.currentTimeMillis();
//        Log.d(TAG, tid + "cpu takes " +  (t4-t3)  + "ms");
//
//        t3 = System.currentTimeMillis();
//        tflite_gpu.run(inputArray, outputArray);
//        t4 = System.currentTimeMillis();
//        Log.d(TAG, tid + "gpu takes " +  (t4-t3)  + "ms");
//
//        tflite_cpu.close();
//        tflite_gpu.close();

//         here start partition inference
        int i = 0 ;
        while(i < models.length){
            String[] parts = models[i].split("_");
            if(parts.length < 3){ // not the cutted model file
                i++;
                continue;
            }

            // Extract the cut number
            int cutNumber = Integer.parseInt(parts[2]);

            // Extract the last three numbers
            int lastNumber1 = Integer.parseInt(parts[parts.length - 3]);
            int lastNumber2 = Integer.parseInt(parts[parts.length - 2]);
            int lastNumber3 = Integer.parseInt(parts[parts.length - 1].replace(".tflite", ""));
            Log.d(TAG,"here is cut " + cutNumber);
//            Log.d(TAG,"here is  " + cutNumber + " " + lastNumber1 + " " + lastNumber2 + " " + lastNumber3);

            float[][][][] middleArray = new float[1][lastNumber1][lastNumber2][lastNumber3];

            Interpreter tflite_cpu = new Interpreter(FileUtil.loadMappedFile(Inference.this,models[i]));
            Interpreter tflite_cpu2 = new Interpreter(FileUtil.loadMappedFile(Inference.this,models[i+1]));

            Interpreter tflite_gpu = new Interpreter(FileUtil.loadMappedFile(Inference.this,models[i]), gpu_options);
            Interpreter tflite_gpu2 = new Interpreter(FileUtil.loadMappedFile(Inference.this,models[i+1]), gpu_options);

            // on big core
            CommandExecution.easyExec("taskset -p f0 " + tid,true);

            // first time
            long t3 = System.currentTimeMillis();
            tflite_cpu.run(inputArray, middleArray);
            long t4 = System.currentTimeMillis();
            Log.d(TAG, "big core left part first time " + (t4 - t3) + " ms");

            long t5 = System.currentTimeMillis();
            tflite_cpu2.run(middleArray, outputArray);
            long t6 = System.currentTimeMillis();
            Log.d(TAG, "big core right part first time " + (t6 - t5) + " ms");

            Log.d(TAG, "big core all first time " + (t6 -t5 + t4-t3) + " ms");

            // second time
            t3 = System.currentTimeMillis();
            tflite_cpu.run(inputArray, middleArray);
            t4 = System.currentTimeMillis();
            Log.d(TAG, "big core left part second time " + (t4 - t3) + " ms");

            t5 = System.currentTimeMillis();
            tflite_cpu2.run(middleArray, outputArray);
            t6 = System.currentTimeMillis();
            Log.d(TAG, "big core right part second time " + (t6 - t5) + " ms");
            Log.d(TAG, "big core all first time " + (t6 -t5 + t4-t3)  + " ms");

            // on little core
            CommandExecution.easyExec("taskset -p 0f " + tid,true);

            // first time
            t3 = System.currentTimeMillis();
            tflite_cpu.run(inputArray, middleArray);
            t4 = System.currentTimeMillis();
            Log.d(TAG, "little core left part first time " + (t4 - t3) + " ms");

            t5 = System.currentTimeMillis();
            tflite_cpu2.run(middleArray, outputArray);
            t6 = System.currentTimeMillis();
            Log.d(TAG, "little core right part first time " + (t6 - t5) + " ms");

            Log.d(TAG, "little core all first time " + (t6 -t5 + t4-t3)  + " ms");

            // second time
            t3 = System.currentTimeMillis();
            tflite_cpu.run(inputArray, middleArray);
            t4 = System.currentTimeMillis();
            Log.d(TAG, "little core left part second time " + (t4 - t3) + " ms");

            t5 = System.currentTimeMillis();
            tflite_cpu2.run(middleArray, outputArray);
            t6 = System.currentTimeMillis();
            Log.d(TAG, "little core right part second time " + (t6 - t5) + " ms");
            Log.d(TAG, "little core all second time " + (t6 -t5 + t4-t3) + " ms");

            // on gpu with little core
            // first time
            t3 = System.currentTimeMillis();
            tflite_gpu.run(inputArray, middleArray);
            t4 = System.currentTimeMillis();
            Log.d(TAG, "gpu(little) left part first time " + (t4 - t3) + " ms");

            t5 = System.currentTimeMillis();
            tflite_gpu2.run(middleArray, outputArray);
            t6 = System.currentTimeMillis();
            Log.d(TAG, "gpu(little) right part first time " + (t6 - t5) + " ms");

            Log.d(TAG, "gpu(little) all first time " + (t6 -t5 + t4-t3) + " ms" );

            // second time
            t3 = System.currentTimeMillis();
            tflite_gpu.run(inputArray, middleArray);
            t4 = System.currentTimeMillis();
            Log.d(TAG, "gpu(little)left part second time " + (t4 - t3) + " ms");

            t5 = System.currentTimeMillis();
            tflite_gpu2.run(middleArray, outputArray);
            t6 = System.currentTimeMillis();
            Log.d(TAG, "gpu(little)right part second time " + (t6 - t5) + " ms");
            Log.d(TAG, "gpu(little)all second time " + (t6 -t5 + t4-t3) + " ms");


            // on gpu with big core
            CommandExecution.easyExec("taskset -p f0 " + tid,true);
            t3 = System.currentTimeMillis();
            tflite_gpu.run(inputArray, middleArray);
            t4 = System.currentTimeMillis();
            Log.d(TAG, "gpu(big)left part first time " + (t4 - t3) + " ms");

            t5 = System.currentTimeMillis();
            tflite_gpu2.run(middleArray, outputArray);
            t6 = System.currentTimeMillis();
            Log.d(TAG, "gpu(big)right part first time " + (t6 - t5) + " ms");

            Log.d(TAG, "gpu(big)all first time " + (t6 -t5 + t4-t3) + " ms" );

            // second time
            t3 = System.currentTimeMillis();
            tflite_gpu.run(inputArray, middleArray);
            t4 = System.currentTimeMillis();
            Log.d(TAG, "gpu(big)left part second time " + (t4 - t3) + " ms");

            t5 = System.currentTimeMillis();
            tflite_gpu2.run(middleArray, outputArray);
            t6 = System.currentTimeMillis();
            Log.d(TAG, "gpu(big)right part second time " + (t6 - t5) + " ms");
            Log.d(TAG, "gpu(big)all second time " + (t6 -t5 + t4-t3)  + " ms");


            // for test only
            t3 = System.currentTimeMillis();
            tflite_cpu.run(inputArray, middleArray);
            tflite_gpu2.run(middleArray, outputArray);
            t4 = System.currentTimeMillis();
            Log.e(TAG, "cpu left and gpu right first " + (t4 - t3) + " ms");

            Log.d(TAG, "one cut finished");

            tflite_cpu.close();
            tflite_cpu2.close();
            tflite_gpu.close();
            tflite_gpu2.close();

            i += 2;
        }
    }

    @Override
    public void onDestroy() {
        if (alertDialog != null && alertDialog.isShowing()) {
            alertDialog.dismiss();
        }
        super.onDestroy();
    }

    private void showToast() {
        Toast.makeText(this, "Service is running", Toast.LENGTH_SHORT).show();
    }
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        showToast();
        String input = intent.getStringExtra("inputExtra");
        createNotificationChannel();
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this,
                0, notificationIntent,  PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Foreground Service")
                .setContentText(input)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentIntent(pendingIntent)
                .build();
        startForeground(1, notification);

        Thread tfliteThread = new Thread(new Runnable() {
            @Override
            public void run() {
                String TAG = "thread one";
                int threadTid = android.os.Process.myTid();
//                CommandExecution.easyExec("taskset -p 0f " + threadTid,true);
//                String output = CommandExecution.execCommand("taskset -p " + threadTid,true).successMsg;
//                Log.d(TAG, output);
                try {
                    TFLiteInference(threadTid);
                } catch (IOException | InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        });
        tfliteThread.start();

//        Thread tfliteThread3 = new Thread(new Runnable() {
//            @Override
//            public void run() {
//                String TAG = "thread two";
//                int threadTid = android.os.Process.myTid();
////                CommandExecution.easyExec("taskset -p f0 " + threadTid,true);
////                String output = CommandExecution.execCommand("taskset -p " + threadTid,true).successMsg;
////                Log.d(TAG, output);
//                try {
//                    TFLiteInference(threadTid);
//                } catch (IOException | InterruptedException e) {
//                    throw new RuntimeException(e);
//                }
//            }
//        });
//        tfliteThread3.start();

        return START_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}