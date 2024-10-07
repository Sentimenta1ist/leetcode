package src.structures;


import java.util.ArrayList;
import java.util.List;

public class Trie {

    public static class TrieNode {
        List<TrieNode> children = new ArrayList<>();
        boolean isKey;
        Character s;

        TrieNode(Character s, boolean isKey){
            this.s = s;
            this.isKey = isKey;
        }
    }

    TrieNode root;

    public Trie() {
        root = new TrieNode(null, false);
    }

    private int lastNode(String word, TrieNode curr) {
        return 0;
    }

    public void insert(String word) {
        int index = 0;


        TrieNode curr = root;

        while (true) {
            boolean noInChild = true;
            for(TrieNode trieNode : curr.children) {
                if(index < word.length() && word.charAt(index) == trieNode.s) {
                    noInChild = false;
                    curr = trieNode;
                    index++;
                    break;
                }
            }
            if(noInChild) {
                break;
            }
        }

        while(index < word.length()) {
            TrieNode trieNode = new TrieNode(word.charAt(index), false);
            curr.children.add(trieNode);
            curr = trieNode;
            index++;
        }
        curr.isKey = true;
    }

    public boolean search(String word) {
        TrieNode curr = root;
        int index = 0;
        while (true) {
            boolean noInChild = true;
            for (TrieNode trieNode : curr.children) {
                if (index < word.length() && word.charAt(index) == trieNode.s ) {
                    noInChild = false;
                    curr = trieNode;
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
        TrieNode curr = root;
        int index = 0;
        while (true) {
            boolean noInChild = true;
            for (TrieNode trieNode : curr.children) {
                if (index < prefix.length() && prefix.charAt(index) == trieNode.s ) {
                    noInChild = false;
                    curr = trieNode;
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
    private void printNode(TrieNode trieNode, String indent) {
        if (trieNode == null) {
            return;
        }

        // Print the current node
        System.out.print(indent); // Print the current indentation
        if (trieNode.s != null) {
            System.out.print("Character: " + trieNode.s);
        } else {
            System.out.print("Root");
        }

        // Check if it's a key
        if (trieNode.isKey) {
            System.out.print(" (Key)");
        }

        System.out.println(); // Move to the next line

        // Print the children with increased indentation
        for (TrieNode child : trieNode.children) {
            printNode(child, indent + "  "); // Increase indentation for children
        }
    }
}
