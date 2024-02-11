"""
Block Builder challenge: Block Building Algorithm based on Murch's proposal

Improves transaction selection by clustering based on shared ancestors and descendants, 
optimizing for higher fee rates. 
For detailed discussion and original proposal by Murch: 
https://lists.linuxfoundation.org/pipermail/bitcoin-dev/2021-May/019020.html

Implementation insights and examples by Murch: 
https://gist.github.com/murchandamus/5cb413fe9f26dbce57abfd344ebbfaf2
"""


import heapq


class CandidateSetBlockbuilder:
    def __init__(self, mempool, weightLimit=4000000):
        self.mempool = mempool  # Reference to the mempool containing all transactions
        self.refMempool = Mempool()  # Create a reference mempool for operations
        self.refMempool.fromDict(
            mempool.txs
        )  # Initialize reference mempool with transactions
        self.selectedTxs = []  # List of transactions selected for block construction
        self.txsToBeClustered = {}  # Transactions pending clustering
        self.clusters = {}  # Stores clusters of transactions
        self.clusterHeap = []  # Priority queue for managing clusters
        self.txClusterMap = {}  # Maps transactions to their respective clusters
        # Set the weight limit for the block, accounting for headers and coinbase
        self.weightLimit = weightLimit
        self.availableWeight = (
            self.weightLimit
        )  # Tracks remaining weight available for transactions

    def cluster(self, weightLimit):
        # Initialize clustering if not yet started
        if len(self.clusters) == 0:
            for txid, tx in self.mempool.txs.items():
                self.txsToBeClustered[txid] = (
                    tx  # Add all mempool transactions to be clustered
                )
        # Iterate over transactions to be clustered
        for txid, tx in self.txsToBeClustered.items():
            if txid in self.txClusterMap:  # Skip if transaction is already clustered
                continue
            localCluster = Cluster(
                tx, weightLimit
            )  # Create a new cluster for the transaction
            localClusterTxids = (
                tx.getLocalClusterTxids()
            )  # Get transaction IDs for the local cluster
            # Add transactions to the cluster
            while len(localClusterTxids) > 0:
                nextTxid = localClusterTxids.pop()
                if (
                    nextTxid in localCluster.txs
                ):  # Skip if transaction is already in the cluster
                    continue
                nextTx = self.mempool.getTx(
                    nextTxid
                )  # Retrieve transaction from mempool
                localCluster.addTx(nextTx)  # Add transaction to the cluster
                localClusterTxids += (
                    nextTx.getLocalClusterTxids()
                )  # Add new transaction IDs to process
            # Map transactions to the cluster and add the cluster to the heap
            self.clusters[localCluster.representative] = localCluster
            for lct in localCluster.txs:
                self.txClusterMap[lct] = localCluster.representative
            heapq.heappush(
                self.clusterHeap, localCluster
            )  # Add cluster to the priority queue

        self.txsToBeClustered = {}  # Reset transactions to be clustered
        return self.clusters  # Return the formed clusters

    def popBestCandidateSet(self, weightLimit):
        # Cluster transactions to prepare for selecting the best candidate set
        self.cluster(weightLimit)
        # Attempt to pop the best (top-priority) cluster from the heap
        bestCluster = heapq.heappop(self.clusterHeap) if len(self.clusterHeap) else None
        # Initialize the best candidate set from the best cluster, if available
        bestCandidateSet = (
            bestCluster.bestCandidate if bestCluster is not None else None
        )
        # If the heap is empty, try to get the best candidate set directly from the cluster
        if 0 == len(self.clusterHeap):
            bestCandidateSet = (
                bestCluster.getBestCandidateSet(weightLimit)
                if bestCluster is not None
                else None
            )
        # Loop to find a candidate set that doesn't exceed the weight limit
        while (
            bestCandidateSet is None or bestCandidateSet.get_weight() > weightLimit
        ) and len(self.clusterHeap) > 0:
            # Refresh the best cluster with a new candidate set considering the weight limit
            if bestCluster.getBestCandidateSet(weightLimit) is None:
                # If no suitable candidate set is found, move to the next cluster
                bestCluster = heapq.heappop(self.clusterHeap)
            else:
                # Re-evaluate the best cluster and possibly find a better candidate set
                bestCluster = heapq.heappushpop(self.clusterHeap, bestCluster)
            bestCandidateSet = bestCluster.bestCandidate

        # Discard the candidate set if it exceeds the weight limit
        if bestCandidateSet is not None and bestCandidateSet.get_weight() > weightLimit:
            bestCandidateSet = None

        # If a suitable candidate set is found, proceed with cleanup
        if bestCandidateSet is not None:
            # Remove references to the candidate set from the cluster
            bestCluster.removeCandidateSetLinks(bestCandidateSet)
            # Clean up the transaction to cluster mappings
            for txid, tx in bestCluster.txs.items():
                self.txClusterMap.pop(txid)
                if txid in bestCandidateSet.txs:
                    # Remove transactions in the candidate set from the mempool
                    self.mempool.txs.pop(txid)
                else:
                    # Stage other transactions for clustering in the next round
                    self.txsToBeClustered[txid] = tx
            # Remove the cluster itself for re-evaluation in future rounds
            self.clusters.pop(bestCluster.representative)

        return bestCandidateSet

    def buildBlockTemplate(self):
        # Stage all transactions in the mempool for clustering
        for txid, tx in self.mempool.txs.items():
            self.txsToBeClustered[txid] = tx
        # Organize transactions into clusters based on the block's weight limit
        self.cluster(self.weightLimit)

        # Continuously attempt to fill the block with transactions until constraints are met
        while len(self.mempool.txs) > 0 and self.availableWeight > 0:
            # Pop the best set of transactions that fit within the remaining weight limit
            bestCandidateSet = self.popBestCandidateSet(self.availableWeight)
            # If no suitable set is found, or the set is empty, stop the process
            if bestCandidateSet is None or len(bestCandidateSet.txs) == 0:
                break
            # Get transactions from the candidate set in topological order for block inclusion
            txsIdsToAdd = bestCandidateSet.get_topologically_sorted_txids()
            # Extend the block with these transactions
            self.selectedTxs.extend(txsIdsToAdd)
            # Decrease the available weight by the weight of transactions added to the block
            self.availableWeight -= bestCandidateSet.get_weight()

        # Return the list of transaction IDs selected for the block
        return self.selectedTxs

    def generate_block_template(self):
        # Construct the file path with a streamlined approach
        file_path = f"./solution/block.txt"
        with open(file_path, "w") as file:

            # Directly proceed to write transaction IDs if any are selected
            for tx_id in self.selectedTxs:
                file.write(f"{tx_id}\n")




class Mempool:
    def __init__(self):
        self.txs = {}

    def fromDict(self, txDict):
        for txid, tx in txDict.items():
            self.txs[txid] = tx

    def load_transactions_from_csv(
        self, file_path, backfill=True, confirmed_transactions={}, delimiter=" "
    ):
        """
        Loads transactions from a CSV file into the mempool, with an option to backfill transaction relatives.

        Args:
        - file_path (str): The path to the CSV file containing transaction data.
        - backfill (bool): Whether to perform backfilling of transaction relatives after loading.
        - confirmed_transactions (dict): A dictionary containing transactions that have already been confirmed.
        - delimiter (str): The delimiter used within the CSV file to separate transaction details.

        The expected CSV format is: transaction_id, value, fee, [parents...]
        Where [parents...] are optional and represent parent transactions of the current transaction.
        """
        with open(file_path, "r") as file:
            # Iterate through each transaction line in the CSV file
            for line in file:
                # Clean the line and split it into transaction elements
                elements = line.strip().split(",")
                # Ensure the line contains at least the minimum transaction data
                if not elements:
                    continue  # Skip empty or malformed lines

                # The first element is always the transaction ID
                transaction_id = elements[0]

                # Ensure there's enough information to construct a transaction
                if len(elements) >= 3:
                    # Parse parent transaction IDs if provided
                    parents = (
                        elements[3].split(";")
                        if len(elements) > 3 and elements[3]
                        else []
                    )
                    # Construct the transaction object with the given data
                    transaction = Transaction(
                        transaction_id, int(elements[1]), int(elements[2]), parents
                    )
                    # Add this transaction to the mempool, keyed by its transaction ID
                    self.txs[transaction_id] = transaction

        # After all transactions are loaded, optionally backfill relatives if requested
        if backfill:
            self.backfill_relatives(confirmed_transactions)

    def get_backfilled_ancestors(
        self, transaction, backfilled_transactions, confirmed_transactions=None
    ):
        """
        Recursively backfills ancestors of a given transaction if not already backfilled.
        It updates the transaction's parents and ancestors sets based on confirmed transactions and availability in mempool.

        Args:
            transaction: The transaction object to backfill ancestors for.
            backfilled_transactions: A set of transaction IDs that have already been backfilled.
            confirmed_transactions: A dictionary of confirmed transactions, defaulting to an empty dict if none provided.
        Returns:
            A set of ancestors for the given transaction after backfilling.
        """
        # Avoid mutable default argument issue
        if confirmed_transactions is None:
            confirmed_transactions = {}

        # Only proceed if the transaction has not been backfilled yet
        if transaction.txid not in backfilled_transactions:
            # Combine potential ancestors (from both parents and previously identified ancestors)
            potential_ancestors = transaction.parents.union(transaction.ancestors)
            transaction.parents = set(potential_ancestors)
            transaction.ancestors = set()

            for ancestor_id in potential_ancestors:
                if ancestor_id in confirmed_transactions:
                    # Exclude confirmed ancestors from further processing
                    transaction.parents.remove(ancestor_id)
                elif ancestor_id not in self.txs:
                    # Handle case where an ancestor is neither confirmed nor present in mempool
                    raise Exception(
                        f"{ancestor_id} is not confirmed and not in mempool"
                    )

                # Continue tracking unconfirmed ancestors
                transaction.ancestors.add(ancestor_id)
                # Recursively backfill further ancestors
                further_ancestors = self.get_backfilled_ancestors(
                    self.txs[ancestor_id],
                    backfilled_transactions,
                    confirmed_transactions,
                )

                # Update the sets of parents and ancestors after backfilling
                transaction.parents.difference_update(further_ancestors)
                transaction.ancestors.update(further_ancestors)

            # Update linkage for ancestors and descendants based on backfilled information
            for ancestor_id in transaction.ancestors:
                self.txs[ancestor_id].descendants.add(transaction.txid)
            for parent_id in transaction.parents:
                self.txs[parent_id].children.add(transaction.txid)

            # Mark this transaction as backfilled
            backfilled_transactions.add(transaction.txid)

        return transaction.ancestors

    def backfill_relatives(self, confirmed_transactions=None):
        if confirmed_transactions is None:
            confirmed_transactions = {}

        # A set to keep track of transactions that have been backfilled
        backfilled_transactions = set()

        for transaction in self.txs.values():
            if transaction.txid not in backfilled_transactions:
                self.get_backfilled_ancestors(
                    transaction, backfilled_transactions, confirmed_transactions
                )
                # After backfill, add transaction to backfilled set
                backfilled_transactions.add(transaction.txid)

            # Check and ensure consistency in ancestor count
            if len(transaction.ancestors) < len(transaction.parents):
                raise Exception("Incomplete ancestor set: fewer ancestors than parents")

    def getTx(self, txid):
        return self.txs[txid]


class Transaction:
    def __init__(
        self,
        txid,
        fee,
        weight,
        parents=None,
        children=None,
        ancestors=None,
        descendants=None,
    ):
        self.txid = txid
        self.fee = int(fee)
        self.feerate = None
        self.weight = int(weight)
        if parents is None:
            parents = []
        self.parents = set([] + parents)
        if ancestors is None:
            ancestors = []
        self.ancestors = set([] + ancestors)
        if children is None:
            children = []
        self.children = set([] + children)
        if descendants is None:
            descendants = []
        self.descendants = set([] + descendants)

    def get_feerate(self):
        if not self.feerate:
            self.feerate = self.fee / self.weight
        return self.feerate

    def getLocalClusterTxids(self):
        return list(set([self.txid] + list(self.children) + list(self.parents)))

    def __lt__(self, other):
        # Sort highest feerate first, use highest weight as tiebreaker
        if self.get_feerate() == other.get_feerate():
            return self.weight > other.weight
        return self.get_feerate() > other.get_feerate()


class TransactionSet:
    def __init__(self, txs):
        self.txs = txs
        self.weight = -1
        self.feerate = -1

    def get_weight(self):
        if self.weight < 0:
            self.weight = sum(tx.weight for tx in self.txs.values())
        return self.weight

    def get_fees(self):
        return sum(tx.fee for tx in self.txs.values())

    def get_feerate(self):
        if self.feerate < 0:
            self.feerate = self.get_fees() / self.get_weight()
        return self.feerate

    def get_topologically_sorted_txids(self):
        return [
            tx.txid
            for tx in sorted(
                list(self.txs.values()), key=lambda tx: (len(tx.ancestors), tx.txid)
            )
        ]


class Cluster:
    def __init__(self, tx, weightLimit):
        self.representative = tx.txid
        self.txs = {tx.txid: tx}
        self.ancestorSets = None
        self.bestCandidate = None
        self.bestFeerate = tx.get_feerate()
        self.weightLimit = weightLimit
        self.eligibleTxs = {tx.txid: tx}
        self.uselessTxs = {}

    def addTx(self, tx):
        self.txs[tx.txid] = tx
        self.eligibleTxs[tx.txid] = tx
        self.representative = min(tx.txid, self.representative)
        self.bestFeerate = max(self.bestFeerate, tx.get_feerate())

    def __lt__(self, other):
        if self.bestFeerate == other.bestFeerate:
            if other.bestCandidate is None:
                return False
            if self.bestCandidate is None:
                return True
            return self.bestCandidate.get_weight() > other.bestCandidate.get_weight()
        return self.bestFeerate > other.bestFeerate

    # Return CandidateSet composed of txid and its ancestors
    def assembleAncestry(self, txid):
        # Initialize ancestor cache if not present
        if self.ancestorSets is None or txid not in self.ancestorSets:
            # Initialize ancestry with target transaction
            tx = self.txs[txid]
            ancestry = {txid: tx}
            # Prepare to explore all parent transactions
            searchList = [] + list(tx.parents)
            # Explore until all ancestors are found
            while len(searchList) > 0:
                ancestorTxid = searchList.pop()
                # Avoid re-adding known ancestors
                if ancestorTxid not in ancestry:
                    ancestor = self.txs[ancestorTxid]
                    ancestry[ancestorTxid] = ancestor
                    # Add parents of the current ancestor for further exploration
                    searchList += ancestor.parents
            # Wrap found ancestors in a CandidateSet
            ancestorSet = CandidateSet(ancestry)
            # Initialize or update the ancestorSets cache
            if self.ancestorSets is None:
                self.ancestorSets = {txid: ancestorSet}
            else:
                self.ancestorSets[txid] = ancestorSet
        # Return the assembled set of ancestors for txid
        return self.ancestorSets[txid]

    def pruneEligibleTxs(self, bestFeerate):
        while 1:
            # Flag to detect changes during iteration
            nothingChanged = True
            # List to hold transactions to be pruned
            prune = []
            # Iterate over all eligible transactions
            for txid, tx in self.eligibleTxs.items():
                # Skip transactions with feerate above threshold
                if tx.get_feerate() >= bestFeerate:
                    continue
                # Prune if no children and feerate is too low
                if len(tx.children) == 0:
                    nothingChanged = False  # Mark change
                    prune.append(txid)  # Add to prune list
                    self.uselessTxs[txid] = tx  # Mark as useless
                # Prune if all children are already marked useless
                elif all(d in self.uselessTxs for d in tx.children):
                    nothingChanged = False  # Mark change
                    prune.append(txid)  # Add to prune list
                    self.uselessTxs[txid] = tx  # Mark as useless
            # Remove pruned transactions from eligible list
            for txid in prune:
                self.eligibleTxs.pop(txid)
            # Exit loop if no changes were made
            if nothingChanged:
                break

    def expandCandidateSet(self, candidateSet, bestFeerate):
        # Gather all direct children of the candidate set
        allChildren = candidateSet.getChildren()
        expandedCandidateSets = []
        for d in allChildren:
            # Skip if child is known to be useless or already in the candidate set
            if d in self.uselessTxs or d in candidateSet.txs:
                continue
            descendant = self.txs[d]
            # Create a new set including the descendant and its ancestors
            expandedSetTxs = {descendant.txid: descendant}
            # Include ancestor transactions of the descendant
            expandedSetTxs.update(self.assembleAncestry(descendant.txid).txs)
            # Combine with the current candidate set transactions
            expandedSetTxs.update(candidateSet.txs)
            # Create a new candidate set with the expanded transactions
            descendantCS = CandidateSet(expandedSetTxs)
            expandedCandidateSets.append(descendantCS)
        return expandedCandidateSets

    def getBestCandidateSet(self, weightLimit):
        # Update weight limit to the minimum of the given limit and existing limit
        self.weightLimit = min(weightLimit, self.weightLimit)
        # Return existing best candidate if it's within the new weight limit
        if (
            self.bestCandidate is not None
            and self.bestCandidate.get_weight() <= self.weightLimit
        ):
            return self.bestCandidate
        # Reset eligible and useless transactions
        self.eligibleTxs = {}
        self.eligibleTxs.update(self.txs)
        self.uselessTxs = {}
        bestCand = None  # Initialize the best candidate set
        previouslyEvaluated = set()  # Keep track of evaluated candidate sets
        searchHeap = []  # Priority queue for candidate sets to evaluate

        # Iterate over all transactions to find the best candidate set
        for txid in self.eligibleTxs.keys():
            cand = self.assembleAncestry(txid)
            # Check if the candidate set meets weight criteria
            if cand.get_weight() <= self.weightLimit:
                # Update best candidate if it has a higher feerate or better weight for the same feerate
                if (
                    bestCand is None
                    or bestCand.get_feerate() < cand.get_feerate()
                    or (
                        bestCand.get_feerate() == cand.get_feerate()
                        and bestCand.get_weight() < cand.get_weight()
                    )
                ):
                    bestCand = cand
                heapq.heappush(
                    searchHeap, bestCand
                )  # Add candidate to the heap for further evaluation

        self.bestCandidate = bestCand
        # Set the best feerate, if a best candidate exists
        if bestCand is not None:
            self.bestFeerate = bestCand.get_feerate()
        else:
            self.bestFeerate = -1

        return self.bestCandidate

    # Used to remove ancestors that got included in block from transactions that remain in the mempool
    def removeCandidateSetLinks(self, candidateSet):
        includedTxids = set(candidateSet.txs.keys())
        remainingTxids = self.txs.keys() - includedTxids
        for txid in remainingTxids:
            tx = self.txs[txid]
            tx.parents = set(tx.parents) - includedTxids
            tx.ancestors = set(tx.ancestors) - includedTxids


class CandidateSet(TransactionSet):
    def __init__(self, txs):
        self.txs = {}  # Initialize transaction dictionary
        self.weight = -1  # Initial weight
        self.feerate = -1  # Initial feerate
        # Ensure transaction set is not empty
        if len(txs) < 1:
            raise TypeError("set cannot be empty")
        # Validate and add transactions where all parents are included in the set
        for txid, tx in txs.items():
            if all(parent in txs for parent in tx.parents):
                self.txs[txid] = tx  # Add transaction to set
            else:
                # Raise error if any parent is missing
                raise TypeError("parent of " + txid + " is not in txs")
        # Initialize base class with validated transactions
        TransactionSet.__init__(self, self.txs)

    def __lt__(self, other):
        # Custom comparison for sorting: prioritize higher feerate; on tie, lower weight
        if self.get_feerate() == other.get_feerate():
            return self.get_weight() > other.get_weight()
        return self.get_feerate() > other.get_feerate()

    def getChildren(self):
        # Generate list of all direct children from transactions in the set
        allChildren = (d for tx in self.txs.values() for d in tx.children)
        # Filter out children already in the set, leaving unexplored ones
        unexploredChildren = set(allChildren) - set(self.txs.keys())
        return list(unexploredChildren)


if __name__ == "__main__":
    mempoolfilepath = "mempool.csv"

    mempool = Mempool()
    mempool.load_transactions_from_csv(mempoolfilepath)
    bb = CandidateSetBlockbuilder(mempool)
    bb.buildBlockTemplate()
    bb.generate_block_template()

    mempool.load_transactions_from_csv(mempoolfilepath)
