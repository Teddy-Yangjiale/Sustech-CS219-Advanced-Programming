import java.util.Random;

public class Dotproduct {
    private static long getElapsedNs(long start, long end) {
        return end - start;
    }

    private static void testInt(int length) {
        int[] v1 = new int[length];
        int[] v2 = new int[length];
        Random rand = new Random(42);

        for (int k = 0; k < length; k++) {
            v1[k] = rand.nextInt(10);
            v2[k] = rand.nextInt(10);
        }

        long start = System.nanoTime();
        int dotProduct = 0;
        for (int k = 0; k < length; k++) {
            dotProduct += v1[k] * v2[k];
        }
        long end = System.nanoTime();

        System.err.println("Result int: " + dotProduct);
        System.out.println(getElapsedNs(start, end));
    }

    private static void testShort(int length) {
        short[] v1 = new short[length];
        short[] v2 = new short[length];
        Random rand = new Random(42);

        for (int k = 0; k < length; k++) {
            v1[k] = (short) rand.nextInt(10);
            v2[k] = (short) rand.nextInt(10);
        }

        long start = System.nanoTime();
        short dotProduct = 0;
        for (int k = 0; k < length; k++) {
            dotProduct += v1[k] * v2[k];
        }
        long end = System.nanoTime();

        System.err.println("Result short: " + dotProduct);
        System.out.println(getElapsedNs(start, end));
    }

    private static void testChar(int length) {
        byte[] v1 = new byte[length];
        byte[] v2 = new byte[length];
        Random rand = new Random(42);

        for (int k = 0; k < length; k++) {
            v1[k] = (byte) rand.nextInt(10);
            v2[k] = (byte) rand.nextInt(10);
        }

        long start = System.nanoTime();
        byte dotProduct = 0;
        for (int k = 0; k < length; k++) {
            dotProduct += v1[k] * v2[k];
        }
        long end = System.nanoTime();

        System.err.println("Result char: " + dotProduct);
        System.out.println(getElapsedNs(start, end));
    }

    private static void testFloat(int length) {
        float[] v1 = new float[length];
        float[] v2 = new float[length];
        Random rand = new Random(42);

        for (int k = 0; k < length; k++) {
            v1[k] = rand.nextInt(10) / 2.0f;
            v2[k] = rand.nextInt(10) / 2.0f;
        }

        long start = System.nanoTime();
        float dotProduct = 0.0f;
        for (int k = 0; k < length; k++) {
            dotProduct += v1[k] * v2[k];
        }
        long end = System.nanoTime();

        System.err.println("Result float: " + dotProduct);
        System.out.println(getElapsedNs(start, end));
    }

    private static void testDouble(int length) {
        double[] v1 = new double[length];
        double[] v2 = new double[length];
        Random rand = new Random(42);

        for (int k = 0; k < length; k++) {
            v1[k] = rand.nextInt(10) / 2.0;
            v2[k] = rand.nextInt(10) / 2.0;
        }

        long start = System.nanoTime();
        double dotProduct = 0.0;
        for (int k = 0; k < length; k++) {
            dotProduct += v1[k] * v2[k];
        }
        long end = System.nanoTime();

        System.err.println("Result double: " + dotProduct);
        System.out.println(getElapsedNs(start, end));
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: java DotProduct <data_type> <vector_length>");
            System.exit(1);
        }

        String dataType = args[0];
        int length;

        try {
            length = Integer.parseInt(args[1]);
        } catch (NumberFormatException ex) {
            System.err.println("Invalid length");
            System.exit(1);
            return;
        }

        switch (dataType) {
            case "int" -> testInt(length);
            case "short" -> testShort(length);
            case "char" -> testChar(length);
            case "float" -> testFloat(length);
            case "double" -> testDouble(length);
            default -> {
                System.err.println("Unknown type");
                System.exit(1);
            }
        }
    }
}
