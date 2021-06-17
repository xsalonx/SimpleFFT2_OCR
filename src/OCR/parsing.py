
def part_for_lines(P: list, Dy=50):
    all_pos = [(p[0], p[1], k) for k, pl in enumerate(P) for p in pl]

    all_pos.sort(key=lambda p: p[0])
    p_rep = all_pos[0]
    L = [[p_rep]]

    for p in all_pos:
        if abs(p[0] - p_rep[0]) <= Dy:
            L[-1].append(p)
        else:
            p_rep = p
            L.append([p_rep])
    for l in L:
        l.sort(key=lambda p: p[1])

    return L


def get_cluster_to_lines_to_plot(P: list, Dy=50):

    L = part_for_lines(P, Dy=Dy)

    Xl = [[x for _, x, _ in l] for l in L]
    Yl = [[y for y, _, _ in l] for l in L]

    return Yl, Xl


def parse_one_line(l, chars_d, Dx, min_same_dist=10):
    prev_x = l[0][1]
    prev_i = l[0][2]
    line_str = chars_d[l[0][2]][0]

    for (_, x, i) in l[1:]:
        c = chars_d[i][0]
        if abs(x - prev_x) <= (Dx[i] + min_same_dist):
            """
                handling letters collisions
            """
            if abs(prev_x - x) < min_same_dist and\
                    ((c == 'l' and line_str[-1] in ['i', '!'])):
                prev_x = x
                line_str = line_str[:-1] + 'l'

            elif (abs(prev_x - x) < min_same_dist + Dx[i]) and\
                    ((c == 'm' and line_str[-1] == 'n') or
                     (c == 'w' and line_str[-1] == 'v') or
                     (c == 'h' and line_str[-1] in ['l', 'i']) or
                     (c == 'n' and line_str[-1] == 'r')):
                prev_x = x
                prev_i = i
                line_str = line_str[:-1] + c
            elif abs(prev_x - x) < min_same_dist and \
                    ((c == 'n' and line_str[-1] == 'm') or
                     (c == 'n' and line_str[-1] == 'h') or
                     (c == '!' and line_str[-1] in ['l', 'i', 'd', 't', 'f']) or
                     (c == 'i' and line_str[-1] in ['l', 'm', 'n']) or
                     (c == line_str[-1]) or
                     (c == 'q' and line_str[-1] in ['d', 'o', 'p'])):
                pass
            elif abs(prev_x - x) < min_same_dist and \
                    ((c == 'l' and line_str[-1] == 'd') or
                     (c == 'i' and line_str[-1] == 'd')):
                pass
            else:
                line_str += c
                prev_x = x
                prev_i = i

        else:
            line_str += ((" " if c!='.' else '') + c)
            prev_x = x
            prev_i = i

    return line_str


def parse_clusters_to_txt(L, chars_d, Dx):
    Lines_txt = [parse_one_line(l, chars_d, Dx=Dx) for l in L]
    return "\n".join(Lines_txt)

def parse(P, chars_d, Dx, Dy=50):

    L = part_for_lines(P, Dy)
    text = parse_clusters_to_txt(L, chars_d=chars_d, Dx=Dx)

    return text
