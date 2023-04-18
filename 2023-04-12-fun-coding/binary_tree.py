from wasabi import msg


def variation_1():
    problem = """
    Given the root of a binary tree, determine if it is a valid binary search
    tree (BST). A valid BST is defined as follows:

    - The left subtree of a node contains only nodes with keys less than the
    node's key.
    - The right subtree of a node contains only nodes with keys greater than 
    the node's key.
    - Both left and right subtrees must also be binary search trees.
    """
    print(problem)

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def solution(root: TreeNode) -> bool:
        # Do a depth-first search first, and then use
        # -inf and +inf as your bounds. If you go left, then
        # the upper bound must be the node, and the lower bound is -inf
        # and so on.

        def dfs(lower, upper, node):
            if not node:
                return True
            elif node.val <= lower or node.val >= upper:
                return False
            else:
                return dfs(lower, node.val, node.left) and dfs(
                    node.val, upper, node.right
                )

        return dfs(float("-inf"), float("inf"), root)


if __name__ == "__main__":
    variation_1()
    msg.divider()
