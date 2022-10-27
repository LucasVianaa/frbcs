import random
from functools import reduce
from statistics import mean

class Func:
    def __init__(self, typ):
        self.opt = ""
        self.type = typ
        self.mid = 0
        self.bottom = 0
        self.top = 0
    
    def value(self, value):
        if self.type == "Low":
            if (value <= self.mid and self.opt == "start") or (value >= self.mid and self.opt == "end"):
                return 1.0
        if value < self.bottom or value > self.top:
            return 0.0
        if value >= self.bottom and value <= self.mid:
            return (value - self.bottom) / (self.mid - self.bottom)
        elif value <= self.top and value >= self.mid:
            return (self.top - value) / (self.top - self.mid)
    
    def get(self):
        return {
            "mid": self.mid,
            "top": self.top,
            "bottom": self.bottom
        }

class Feature:
    def __init__(self):
        self.functions = {}

    def get(self):
        print_dict = {
            "Low": [],
            "Medium": [],
            "High": []
        }

        for key, value_list in self.functions.items():
            if key in print_dict:
                print_dict[key] = [x.get() for x in value_list]

        return print_dict
    
    def value(self, value):
        returned_dict = {
            "Low": 0.0,
            "Medium": 0.0,
            "High": 0.0
        }

        for key, value_list in self.functions.items():
            if key in returned_dict:
                returned_dict[key] = max([x.value(value) for x in value_list])
            else:
                returned_dict[key] = 0.0
        #max_key = max(returned_dict, key=returned_dict.get)
        return returned_dict
    
class Category:
    def __init__(self):
        self.features = {}

    def get(self):
        printed = {}
        for key, value in self.features.items():
            printed[key] = self.features[key].get()
        return printed
    
    def value(self, value_list):
        return_list = {}
        for key, value in value_list.items():
            return_list[key] = self.features[key].value(value)

        return return_list

class Rule:
    def __init__(self, antecedent, label):
        self.antecedent = antecedent
        self.label = label
        self.weight = 0.0
    
    def compare(self, rule, label=True):
        other_antecedent = rule.antecedent
        if not self.label == rule.label and label:
            return False
        for feature in self.antecedent:
            max_key_other = max(other_antecedent[feature], key=other_antecedent[feature].get)
            max_key_self = max(self.antecedent[feature], key=self.antecedent[feature].get)
            if not max_key_other == max_key_self:
                return False
        return True
        
    def __str__(self) -> str:
        start = True
        parsed = "If "
        for key in self.antecedent:
            max_key = max(self.antecedent[key], key=self.antecedent[key].get)
            if not start:
                parsed += " and "
            parsed += key + " is " + max_key
            start = False
        parsed += ", then the element is a " + self.label + " with weight " + str(self.weight)
        return parsed

class Rulebase:
    def __init__(self):
        self.rules = []
        self.trained = False
    
    def generate_weights(self, dataset):
        numerator = 0.0
        denominator = 0.0
        total_rules = []
        for rule in self.rules:
            for item in dataset.train:
                for feature in item.antecedent:
                    max_key_rule = max(rule.antecedent[feature], key=rule.antecedent[feature].get)
                    total_rules.append(item.antecedent[feature][max_key_rule])
                total = reduce((lambda x, y: x * y), total_rules)
                denominator += total
                if item.label == rule.label:
                    numerator += total
                total_rules = []
            if denominator > 0.0:
                rule.weight = numerator / denominator
            numerator = 0.0
            denominator = 0.0
        self.trained = True

    def eliminate_bad_rules(self):
        tmp_rules = self.rules.copy()
        to_delete = []
        for i, rule in enumerate(tmp_rules):
            if rule.weight <= 0:
                if i not in to_delete:
                    to_delete.append(i)

        self.rules = [i for j, i in enumerate(self.rules) if j not in to_delete]

    def eliminate_conflicts(self):
        tmp_rules = self.rules.copy()
        to_delete = []
        for i, rule in enumerate(tmp_rules):
            if i + 1 < len(tmp_rules):
                for j, rule_cmp in enumerate(tmp_rules[i+1:]):
                    if rule.compare(rule_cmp, False):
                        if rule.weight > rule_cmp.weight:
                            if i+j+1 not in to_delete:
                                to_delete.append(i+j+1)
                        else:
                            if i not in to_delete:
                                to_delete.append(i)
                            break

        self.rules = [i for j, i in enumerate(self.rules) if j not in to_delete]

    def eliminate_redundant(self):
        tmp_rules = self.rules.copy()
        to_delete = []
        for i, rule in enumerate(tmp_rules):
            if i + 1 < len(tmp_rules):
                for j, rule_cmp in enumerate(tmp_rules[i+1:]):
                    if rule.compare(rule_cmp):
                        if i+j+1 not in to_delete:
                            to_delete.append(i+j+1)
        self.rules = [i for j, i in enumerate(self.rules) if j not in to_delete]


    def estimate(self, item, category):
        antecedent = category.value(item.value)
        total_by_label = {
            "text": [],
            "non-text": []
        }
        total_rules = []
        for rule in self.rules:
            for feature in antecedent:
                max_key_rule = max(rule.antecedent[feature], key=rule.antecedent[feature].get)
                total_rules.append(antecedent[feature][max_key_rule])
            total = reduce((lambda x, y: x * y), total_rules)
            if total > 0:
                total_by_label[rule.label].append(total * rule.weight)
            total_rules = []
        
        if not total_by_label["text"]:
            total_by_label["text"] = [0.0]
        if not total_by_label["non-text"]:
            total_by_label["non-text"] = [0.0]
        
        text_avg = mean(total_by_label["text"])
        non_text_avg = mean(total_by_label["non-text"])

        return "text" if text_avg > non_text_avg else "non-text"

    def test_base(self, dataset, category):
        if not self.trained:
            return -1
        confusion_matrix = [0,0,0,0]
        for item in dataset.test:
            estimate = self.estimate(item, category)
            if estimate == "text" and item.label == "text":
                confusion_matrix[0] += 1
            elif estimate == "text" and item.label == "non-text":
                confusion_matrix[1] += 1
            elif estimate == "non-text" and item.label == "non-text":
                confusion_matrix[2] += 1
            else:
                confusion_matrix[3] += 1
        
        accuracy = (confusion_matrix[0] + confusion_matrix[2]) / sum(confusion_matrix)
        precision = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1])
        recall = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[3])
        f1_score = 2 * (precision * recall) / (precision + recall)

        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(f1_score))

    def __str__(self):
        total = ""
        for rule in self.rules:
            total += str(rule) + '\n'
        return total
    
class Item:
    def __init__(self, label, value):
        self.label = label
        self.value = value
        self.antecedent = None

class Dataset:
    def __init__(self):
        self.total = []
        self.train = []
        self.test = []
    
    def divide(self, train, test):
        if not train + test == 1.0:
            return "Train + test tem que ser igual a 1.0"

        by_labels = {}
        for item in self.total:
            if item.label in by_labels:
                by_labels[item.label].append(item)
            else:
                by_labels[item.label] = [item]
        for label in by_labels:
            self.train += random.sample(by_labels[label], int(len(by_labels[label])*train))
            self.test += random.sample(by_labels[label], int(len(by_labels[label])*test))
    
    def generate_rules(self, category):
        rulebase = Rulebase()
        for item in self.train:
            antecedent = category.value(item.value)
            item.antecedent = antecedent
            label = item.label
            rulebase.rules.append(Rule(antecedent, label))
        return rulebase




def main():
    non_text = Category()
    text = Category()
    dataset = Dataset()

    # Read functions file
    current = "text"
    current_feature = "height"
    current_func = "Low"
    counter = 0
    with open("functions", "r") as func_file:
        for iter_line in func_file.readlines():
            line = iter_line.replace("\n", '')
            if line.startswith("-"):
                current_feature = line[2:].replace('\n','')
                text.features[current_feature] = Feature()
                non_text.features[current_feature] = Feature()
            elif line.startswith("*"):
                current = line[2:].replace('\n','')
            elif line.startswith(">"):
                counter = 0
                current_func = line[2:].replace('\n','')
                new_fn = Func(current_func)
                if current == "text":
                    text.features[current_feature].functions[current_func] = [new_fn]
                else:
                    non_text.features[current_feature].functions[current_func] = [new_fn]
            else:
                if not counter == 0:
                    if current == "text":
                        text.features[current_feature].functions[current_func].append(Func(current_func))
                    else:
                        non_text.features[current_feature].functions[current_func].append(Func(current_func))

                splitted_line = line.split()
                bottom = float(splitted_line[0])
                mid = float(splitted_line[1])
                top = float(splitted_line[2])
                opt = splitted_line[3]
                if current == "text":
                    text.features[current_feature].functions[current_func][-1].bottom = bottom
                    text.features[current_feature].functions[current_func][-1].top = top
                    text.features[current_feature].functions[current_func][-1].mid = mid
                    text.features[current_feature].functions[current_func][-1].opt = opt
                else:
                    non_text.features[current_feature].functions[current_func][-1].bottom = bottom
                    non_text.features[current_feature].functions[current_func][-1].top = top
                    non_text.features[current_feature].functions[current_func][-1].mid = mid
                    non_text.features[current_feature].functions[current_func][-1].opt = opt
                counter += 1
    

    # Read dataset
    with open("page-blocks.data", "r") as data_file:
        for work_line in data_file.readlines():
            line = work_line.replace('\n', '').split()
            new_value = {
                "height": float(line[0]),
                "length": float(line[1]),
                "area": float(line[2]),
                "eccen": float(line[3]),
                "p_black": float(line[4]),
                "p_and": float(line[5]),
                "mean_tr": float(line[6]),
                "blackpix": float(line[7]),
                "blackand": float(line[8]),
                "wb_trans": float(line[9])
            }
            label = "text" if line[10] == '1' else "non-text"
            dataset.total.append(Item(label, new_value))
    
    # Divide dataset 80-20 train/test
    dataset.divide(0.80, 0.20)
    
    # Generate initial rules
    rulebase = dataset.generate_rules(text)

    # Eliminate redundancies before generating the weights
    rulebase.eliminate_redundant()

    # Generate weights for each rule
    rulebase.generate_weights(dataset)

    # Eliminate unnecessary rules 
    rulebase.eliminate_bad_rules()
    rulebase.eliminate_conflicts()

    # Test database - Prints accuracy, precision, recall and f1-score
    rulebase.test_base(dataset, text)

    # Prints the rules database
    print(rulebase)

    # Estimate value example
    '''
     new_value1 = {
            "height": 10,
            "length": 37,
            "area": 370,
            "eccen": 3.7,
            "p_black": 0.154,
            "p_and": 0.457,
            "mean_tr": 1.39,
            "blackpix": 57,
            "blackand": 169,
            "wb_trans": 41
        }
    item1 = Item("text", new_value1)

    print(rulebase.estimate(category=text, item=item1))

    new_value2 = {
            "height": 25,
            "length": 1,
            "area": 25,
            "eccen": 0.04,
            "p_black": 1.0,
            "p_and": 1.0,
            "mean_tr": 25.0,
            "blackpix": 25,
            "blackand": 25,
            "wb_trans": 1
        }
    item2 = Item("text", new_value2)

    print(rulebase.estimate(category=text, item=item2))
    '''

main()