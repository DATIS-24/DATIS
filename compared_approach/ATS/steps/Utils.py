class ATSUtils(object):

    def get_p_q_list(self, n, i):
        # if n==2:
        #     num_list = list(range(n))
        #     pq_list = []
        #     for p in num_list:
        #         for q in num_list:
        #             if p != q:
        #                 pq_list.append((p, q))
        #     return pq_list
        # else:
            num_list = list(range(n))
            num_list.remove(i)
            import itertools
            pq_list = []
            for pq in itertools.combinations(num_list, 2):
                pq_list.append(pq)
            return pq_list
    

        
