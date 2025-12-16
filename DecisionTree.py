import numpy as np

# Imagine we drop all of our training examples into one big room (the root node).
# The tree's job is to keep asking smart yes/no questions like:
#   "Is feature j <= threshold t?"
# Each good question splits the room into two smaller rooms where the labels are
# more "pure" (less mixed). We measure "how mixed" a room is using entropy.
# The best question is the one that reduces entropy the most (information gain).
# We repeat this recursively until we hit a stopping rule, then we create a leaf.

# entropy is used to measure the impurity of a set of labels
def entropy(y: np.ndarray) -> float:
    """
    Compute the entropy of a binary label array.
    """
    # If a room is empty, it has no uncertainty
    if y.size == 0:
        return 0.0
    # Flatten to a simple 1D list of labels
    y = np.asarray(y).ravel()

     # For binary labels {0,1}, mean(y) is the fraction of class 1
    p_1 = np.mean(y)
    p_0 = 1 - p_1

    # If everyone in the room is the same class, the room is perfectly pure
    if p_1 == 0 or p_0 == 0:
        return 0.0
    # Otherwise, compute entropy: how "mixed" this room is
    return - (p_1 * np.log2(p_1) + p_0 * np.log2(p_0))

# Split dataset based on feature and threshold
def split_dataset(X, node_indices, feature, threshold):
    """
    Split dataset indices into left and right child nodes based on a feature and threshold.
    """
    # We only look at the examples currently living in THIS node
    X_node = X[node_indices, feature]

    # Our yes/no question:
    #   Left room  = feature value <= threshold
    #   Right room = feature value  > threshold
    left_mask = X_node <= threshold

    # Return *indices* for left and right rooms (we keep data in-place, just move pointers)
    return node_indices[left_mask], node_indices[~left_mask]

# Compute information gain for a split, measured by reduction in entropy
def compute_information_gain(X, y, node_indices, feature, threshold):
    """
    Compute the information gain of a split on a given feature and threshold.
    """
    # First, try the split and see who goes left vs right
    left_idx, right_idx = split_dataset(X, node_indices, feature, threshold)

    # If our question sends everyone to one side, it's a useless (degenerate) question
    if len(left_idx) == 0 or len(right_idx) == 0:
        return 0.0
    
    # Labels in the current room, and in each child room
    y_node = y[node_indices]
    y_left = y[left_idx]
    y_right = y[right_idx]

    # Entropy before splitting (how mixed the parent room is)
    H_node = entropy(y_node)

    # Entropy after splitting (how mixed each child room is)
    H_left = entropy(y_left)
    H_right = entropy(y_right)

    # Weight each child entropy by how many samples it contains
    w_left = len(left_idx) / len(node_indices)
    w_right = len(right_idx) / len(node_indices)

    weighted_entropy = w_left * H_left + w_right * H_right

    # Information gain = uncertainty before - uncertainty after
    # (bigger is better -> we want the biggest impurity reduction)
    return H_node - weighted_entropy

# Find the best feature and threshold to split on
def get_best_split(X, y, node_indices, feature_indices=None, verbose=False):
    """
    Determine the best feature and threshold to split on based on information gain.
    """
    # If we weren't told which features to consider, we do a classic trick:
    # check only sqrt(n_features) random features (common in tree/forest style).
    if feature_indices is None:
        n_features = X.shape[1]
        num_features_to_check = int(np.sqrt(n_features))
        num_features_to_check = max(1, num_features_to_check)
        feature_indices = np.random.choice(n_features, num_features_to_check, replace=False)

    # Track the best question found so far
    best_feature = -1
    best_threshold = None
    best_info_gain = -1.0

    # for each candidate feature, we try possible thresholds and keep the best
    for i, feature in enumerate(feature_indices):
        # look at feature values for samples inside this node only
        values = X[node_indices, feature]
        unique_values = np.unique(values)
        
        # if there's no variety in this feature here, it can't split anything
        if len(unique_values) < 2:
            continue
            
        # if the feature has too many unique values, scanning all thresholds is slow
        # so we sample a few thresholds using percentiles
        if len(unique_values) > 20:
            percentiles = np.linspace(0, 100, 22)[1:-1]
            thresholds = np.unique(np.percentile(unique_values, percentiles))
        else:
            # otherwise, try thresholds directly from unique values (expect the last)
            thresholds = unique_values[:-1]

        # Test each threshold: "Is feature <= threshold?"
        for threshold in thresholds:
            info_gain = compute_information_gain(X, y, node_indices, feature, threshold)
            
            # Keep the best question so far
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_threshold = threshold

        # Optional progress log (mostly useful for huge feature sets)
        if verbose and (i + 1) % 1000 == 0:
            print(f"Evaluated {i+1} features so far...")

    # If best_feature stays -1, we failed to find any useful split
    return best_feature, best_threshold

# Decision Tree Implementation
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, n_samples=None):
        # If this is an internal node, it stores the question: feature + threshold
        self.feature = feature
        self.threshold = threshold
        # Pointers to child rooms
        self.left = left
        self.right = right
        # If this is a leaf, value stores the model output (here: prob of class 1)
        self.value = value
        # How many training samples reached this node (useful for summaries/debugging)
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        # Leaf nodes are the end of the story: no more questions, just an answer
        return self.value is not None

# Decision Tree Class   
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, verbose=False):
        # Stopping rules:
        # - don't grow deeper than max_depth
        # - don't split nodes smaller than min_samples_split
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.verbose = verbose
        self.root = None

    # Return the majority class label from y. (Not used in leaves here, but handy utility.)
    def majority_class(self, y):
        """
        Return the majority class label from y.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def build_tree(self, X, y, node_indices, depth):
        """
        Recursively build the decision tree.
        """
        # This node's labels: the "truth" living in the current room
        y_node = y[node_indices]
        n_node = len(node_indices)

        if self.verbose:
            print(f"[Depth {depth}] Building node with {len(node_indices)} samples")

        # Stopping criteria: when the story ends and we make a leaf
        # We stop if:
        # - we hit max depth
        # - we don't have enough samples to justify splitting
        # - the room is already pure (entropy == 0)
        if (depth >= self.max_depth
            or len(node_indices) < self.min_samples_split
            or entropy(y_node) == 0):

            # Leaf output: probability of class 1 in this room
            # (so predict_proba works naturally; predict will threshold at 0.5)
            prob_class_1 = np.mean(y_node)
            if self.verbose:
                print(f"[Depth {depth}] Leaf with prob {prob_class_1:.2f} "
                      f"(n={len(node_indices)})")
                
            return TreeNode(value=prob_class_1, n_samples=n_node)
        
        # Otherwise, try to find the best question to split this room
        best_feature, best_threshold = get_best_split(X, y, node_indices, feature_indices=None, verbose=self.verbose)
        # If we couldn't find a useful split, make a leaf
        if best_feature == -1:
            prob_class_1 = np.mean(y_node)
            if self.verbose:
                print(f"[Depth {depth}] No useful split, leaf prob {prob_class_1:.2f}")
            return TreeNode(value=prob_class_1, n_samples=n_node)
        # Split the room into two child rooms based on the best question
        left_idx, right_idx = split_dataset(X, node_indices, best_feature, best_threshold)
        
        # safety check: if the split collapses, make a leaf
        if len(left_idx) == 0 or len(right_idx) == 0:
            prob_class_1 = np.mean(y_node)
            if self.verbose:
                print(f"[Depth {depth}] Degenerate split, leaf prob {prob_class_1:.2f}")
            return TreeNode(value=prob_class_1, n_samples=n_node)
        
        # Recursively build the left and right child nodes
        if self.verbose:
            print(f"[Depth {depth}] Split on feature {best_feature} <= {best_threshold:.2f} "
                  f"Left {len(left_idx)}, right {len(right_idx)}")

        # ask more questions in the left and right rooms
        left_child = self.build_tree(X, y, left_idx, depth + 1)
        right_child = self.build_tree(X, y, right_idx, depth + 1)

        # this internal node stores the question and links to its children
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child, n_samples=n_node)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the decision tree to the training data.
        """
        # convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # story begins at the root node with all samples
        n_samples = X.shape[0]
        node_indices = np.arange(n_samples)
        # build the tree recursively from the root down
        self.root = self.build_tree(X, y, node_indices, depth=0)

    def predict_one(self, x):
        # walk down the tree by answering each node's question
        node = self.root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        # at the leaf, we return the stored probability of class 1
        return node.value
    
    def predict_proba_one(self, x):
        # same walk as predict_one; keep separate for clarity/usage symmetry
        node = self.root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input samples X.
        """
        X = np.asarray(X)
        # since leaves store P(class=1), we turn that into a hard label with a 0.5 threshold
        predictions = np.array([1 if self.predict_one(x) >= 0.5 else 0 for x in X])
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input samples X.
        Returns an array of shape (n_samples, 2), where column 1 is prob of class 1.
        """
        X = np.asarray(X)
        # First gather P(class=1) for each sample
        probs_class_1 = np.array([self.predict_one(x) for x in X])
        # Then return [P(class=0), P(class=1)] for each sample
        probs = np.vstack([1 - probs_class_1, probs_class_1]).T
        return probs

def summarize_tree(tree):
    """
    Summarize a fitted DecisionTree:
    - number of leaves
    - max depth
    - average leaf size
    - min / max leaf size
    - rough feature usage counts
    """
    # after the tree has grown, we take a walk through it to describe what it built
    node = tree.root
    leaf_depths = []
    leaf_sizes = []
    feature_counts = {}

    def dfs(node, depth):
        # if we reached a leaf, record where the story ended for this path
        if node.is_leaf:
            leaf_depths.append(depth)
            leaf_sizes.append(node.n_samples if hasattr(node, "n_samples") else None)
            return
        
        # count which features were used to ask questions
        feature_counts[node.feature] = feature_counts.get(node.feature, 0) + 1
        
        # keep walking down both branches
        dfs(node.left, depth + 1)
        dfs(node.right, depth + 1)

    dfs(node, depth=0)

    sizes = [s for s in leaf_sizes if s is not None]

    print("Tree Summary")
    print(f"Max depth: {max(leaf_depths) if leaf_depths else 0}")
    print(f"Number of leaves: {len(leaf_depths)}")
    if sizes:
        print(f"Leaf size: mean={np.mean(sizes):.1f}, min={np.min(sizes)}, max={np.max(sizes)}")

    # show which features the tree relied on most (top 10)
    if feature_counts:
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top features used in splits:")
        for f, c in top_features:
            print(f"  feature {f} used {c} times")