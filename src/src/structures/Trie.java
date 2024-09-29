package src.structures;


import java.util.ArrayList;
import java.util.List;

public class Trie {

    public static class Node {
        List<Node> children = new ArrayList<>();
        boolean isKey;
        Character s;

        Node(Character s, boolean isKey){
            this.s = s;
            this.isKey = isKey;
        }
    }

    Node root;

    public Trie() {
        root = new Node(null, false);
    }

    private int lastNode(String word, Node curr) {
        return 0;
    }

    public void insert(String word) {
        int index = 0;


        Node curr = root;

        while (true) {
            boolean noInChild = true;
            for(Node node : curr.children) {
                if(index < word.length() && word.charAt(index) == node.s) {
                    noInChild = false;
                    curr = node;
                    index++;
                    break;
                }
            }
            if(noInChild) {
                break;
            }
        }

        while(index < word.length()) {
            Node node = new Node(word.charAt(index), false);
            curr.children.add(node);
            curr = node;
            index++;
        }
        curr.isKey = true;
    }

    public boolean search(String word) {
        Node curr = root;
        int index = 0;
        while (true) {
            boolean noInChild = true;
            for (Node node : curr.children) {
                if (index < word.length() && word.charAt(index) == node.s ) {
                    noInChild = false;
                    curr = node;
                    index++;
                    break;
                }
            }
            if(noInChild) {
                break;
            }
        }

        return index == word.length() && curr.isKey;
    }

    public boolean startsWith(String prefix) {
        Node curr = root;
        int index = 0;
        while (true) {
            boolean noInChild = true;
            for (Node node : curr.children) {
                if (index < prefix.length() && prefix.charAt(index) == node.s ) {
                    noInChild = false;
                    curr = node;
                    index++;
                    break;
                }
            }
            if(noInChild) {
                break;
            }
        }

        return index == prefix.length();
    }

    public void print() {
        printNode(root, "");
    }

    // Recursive function to print the tree structure
    private void printNode(Node node, String indent) {
        if (node == null) {
            return;
        }

        // Print the current node
        System.out.print(indent); // Print the current indentation
        if (node.s != null) {
            System.out.print("Character: " + node.s);
        } else {
            System.out.print("Root");
        }

        // Check if it's a key
        if (node.isKey) {
            System.out.print(" (Key)");
        }

        System.out.println(); // Move to the next line

        // Print the children with increased indentation
        for (Node child : node.children) {
            printNode(child, indent + "  "); // Increase indentation for children
        }
    }
}
