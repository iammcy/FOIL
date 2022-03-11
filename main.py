import copy
import re
import math
import numpy as np


class knowledge_base:
    def __init__(self, all_triple: dict, target_relation: str) -> None:
        self.all_triple = all_triple
        self.target_relation = target_relation
        self.hypothesis = dict()
        
        # 构建正例、反例、背景知识
        self.__process()


    def __process(self) -> None:
        self.pos = copy.deepcopy(self.all_triple[self.target_relation])
        self.neg = []
        for key, value in self.all_triple.items():
            if key != self.target_relation:
                self.neg += copy.deepcopy(value)
        self.bg = copy.deepcopy(self.all_triple)
        del self.bg[self.target_relation]
    

    def training(self, var_num: int = 3) -> None:
        # 规则包含变量设置
        self.var_num = var_num

        # 回溯归纳学习
        self.__inductive_learning(len(self.pos), len(self.neg))


    def __inductive_learning(self, old_m_pos: int = None, old_m_neg: int = None) -> bool:

        # 当覆盖正例且不覆盖任意反例时，且包含x和y变量，学习结束
        if old_m_pos > 0 and old_m_neg == 0 :
            tag0 = False
            tag1 = False
            for relation, val in self.hypothesis.items():
                for var_idx in val:
                    if 1 in var_idx:
                        tag1 = True
                    if 0 in var_idx:
                        tag0 = True
            if tag0 and tag1:
                return True

        foil_gain_list = []
        candidate_add_hypothesis = []
        candidate_pos_tag = []
        candidate_neg_tag = []
        
        # 0. 依次尝试剩余谓词
        for relation in self.bg.keys():
            
            # 1. 遍历该谓词的每一种前提情况
            for i in range(self.var_num):
                for j in range(self.var_num):
                    if i == j or (relation in self.hypothesis and [i,j] in self.hypothesis[relation]):
                        continue

                    # 2. 依据背景知识对规则进行实例化推理
                    hypothesis = copy.deepcopy(self.hypothesis)
                    add_hypothesis = {relation: [i, j]}
                    if relation not in hypothesis:
                        hypothesis[relation] = []
                    hypothesis[relation].append(add_hypothesis[relation])
                    ret = self.__reasoning(hypothesis)

                    # 3. 基于正例和反例计算增益值
                    pos_tag = np.zeros(len(self.pos), np.int)
                    neg_tag = np.zeros(len(self.neg), np.int)
                    for r in ret:
                        for idx, pos in enumerate(self.pos):
                            if (r[0] == '' and r[1] == '') or (r[0] == pos[0] and r[1] == '') or (r[0] == '' and r[1] == pos[1]) or (r[0] == pos[0] and r[1] == pos[1]):
                                pos_tag[idx] = 1
                        for idx, neg in enumerate(self.neg):
                            if (r[0] == '' and r[1] == '') or (r[0] == neg[0] and r[1] == '') or (r[0] == '' and r[1] == neg[1]) or (r[0] == neg[0] and r[1] == neg[1]):
                                neg_tag[idx] = 1
                        
                    new_m_pos = pos_tag.sum()
                    new_m_neg = neg_tag.sum()

                    # 3.1 取值NA跳过
                    if new_m_pos == 0:
                        continue
                    
                    # 3.2 计算增益值
                    foil_gain = new_m_pos * (math.log2(new_m_pos / (new_m_pos + new_m_neg)) 
                                            - math.log2(old_m_pos / (old_m_pos + old_m_neg)))

                    # 4. 将大于等于0增益值的谓词加入候选
                    foil_gain_list.append(foil_gain)
                    candidate_add_hypothesis.append(add_hypothesis)
                    candidate_pos_tag.append(pos_tag)
                    candidate_neg_tag.append(neg_tag)

        # 5. 对增益值排序
        foil_gain_list = np.array(foil_gain_list)
        candidate_sorted_idx = np.argsort(-foil_gain_list)

        # 6. 遍历降序的候选谓词
        for i in candidate_sorted_idx:

            # 7. 加入本轮前提约束谓词
            relation = list(candidate_add_hypothesis[i].keys())[0]
            if relation not in self.hypothesis:
                self.hypothesis[relation] = []
            self.hypothesis[relation].append(candidate_add_hypothesis[i][relation])

            # 8. 将训练样例中与新的推理规则不符的样例去掉
            pos = copy.deepcopy(self.pos)
            neg = copy.deepcopy(self.neg)

            for j in range(len(candidate_pos_tag[i])-1, -1, -1):
                if candidate_pos_tag[i][j] == 0:
                    self.pos.remove(self.pos[j])
            for j in range(len(candidate_neg_tag[i])-1, -1, -1):
                if candidate_neg_tag[i][j] == 0:
                    self.neg.remove(self.neg[j])

            # 9. 进入下一轮学习
            complete = self.__inductive_learning(candidate_pos_tag[i].sum(), candidate_neg_tag[i].sum())

            # 10. 是否已经学习完成
            if complete == True :
                return True

            # 11. 回滚
            self.pos = pos
            self.neg = neg
            if len(self.hypothesis[relation]) == 1:
                del self.hypothesis[relation]
            else:
                self.hypothesis[relation].remove(candidate_add_hypothesis[i][relation])
        
        return False


    def __reasoning(self, hypothesis: dict) -> list:
        """
        Description:
            According to reasoning rules, use background knowledge to reason.

        Args:
            hypothesis: The premise constraint predicates of reasoning rules.
            For example:
            {
                'relation1':[[0,1],[1,2]],
                'relation2':[[0,2]]
            }
            
        Returns:
            a list of reasoning results.
            For example:
            [
                ['David','','Mike'],
                ['','Ann','James']
            ]
        """

        res = []
        variables = [''] * self.var_num
        relations = list(hypothesis.keys())
        self.__dfs(hypothesis, relations, 0, 0, variables, res)
        return res
    

    def __dfs(self, hypothesis: dict, relations: list, rel_idx: int, val_idx: int, variables: list, res: list) -> None:
        rel = relations[rel_idx]
        i, j = hypothesis[rel][val_idx]

        for head, tail in self.bg[rel]:
            if (variables[i] == '' or variables[i] == head) and (variables[j] == '' or variables[j] == tail):
                tmp_i = variables[i]
                tmp_j = variables[j]
                variables[i] = head
                variables[j] = tail
                if rel_idx == len(relations)-1 and val_idx == len(hypothesis[rel])-1:
                    res.append(copy.deepcopy(variables))
                else:
                    if val_idx != len(hypothesis[rel])-1:
                        self.__dfs(hypothesis, relations, rel_idx, val_idx+1, variables, res)
                    else:
                        self.__dfs(hypothesis, relations, rel_idx+1, 0, variables, res)
                variables[i] = tmp_i
                variables[j] = tmp_j

    
    def show_result(self):
        """

        Description:
            to show reasoning rules and reasoning facts

        """

        if len(self.hypothesis) == 0:
            print("There are no rules of reasoning.")
            return

        # 0. 输出推理规则
        var_letter = ['x', 'y', 'z', 'v', 'r']
        h = []
        for relation, val in self.hypothesis.items():
            for i, j in val: 
                h.append(relation + '('+ var_letter[i] + ',' + var_letter[j] +')')
        rule = ' Ʌ '.join(h)
        rule += ' → ' + self.target_relation + '(' + var_letter[0] + ',' + var_letter[1] + ')'
        print(rule)
        
        # 1. 输出推理事实
        ret = self.__reasoning(self.hypothesis)
        for r in ret:
            if r[:2] in self.all_triple[self.target_relation]:
                continue
            print(self.target_relation + '(' + r[0] + ',' + r[1] + ')')
            

def main():

    # 0.输入第一行
    triple_num = int(input())

    # 1.输入知识图谱数据
    all_triples = dict()
    pattern = re.compile(r'[a-zA-Z]+')

    for i in range(triple_num):
        relation, head, tail = re.findall(pattern, input())
        
        if relation not in all_triples:
            all_triples[relation] = []

        all_triples[relation].append([head, tail])

    # 2.输入目标谓词
    target_relation, _, _ = re.findall(pattern, input())

    # 3.构建知识图谱
    KG = knowledge_base(all_triples, target_relation)

    # 4.FOIL
    KG.training()

    # 5.结果输出
    KG.show_result()


if __name__ == '__main__':
    main()
    