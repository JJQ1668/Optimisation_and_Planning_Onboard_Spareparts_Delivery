import os

try:
    table_max_width = os.get_terminal_size().columns
except OSError:
    table_max_width = 100


def print_table(title=None, content=None, line='=', ct=' ', width=0, border='|'):
    print_line(line, width=width)
    if title is None:
        return
    print_line(title, sp=ct, border=border, width=width)
    if content is None:
        print_line('  ', sp=' ', border=border, width=width)
        print_line(title, sp=ct, border=border, width=width)
        return
    for l in content:
        if len(l) == 1:
            print_line(l, width=width)
        else:
            print_line(l, sp=' ', border=border, width=width, ps=-1)
    print_line(line, width=width)
    print('\n\n')
    
def print_tables(titles, print_lines):
    llens = []
    [llens.extend([len(t) for t in tt])for tt in print_lines]
    llens = sorted(llens)
    if llens[-1] <= int(table_max_width / 3) - 4:
        table_width = llens[-1]
    else:
        table_width = int(llens[-int(len(llens) / 6)] / 2) * 2
        table_width = max(table_width, int(llens[-1] / 3) - 4)
        table_width = min(table_width, table_max_width - 4)
    for i,table in enumerate(print_lines):
        print_table(title=titles[i], content=table, ct='+', width=table_width)

# ps = 0 center, -1 left, 1 right
def print_line(content, sp=' ', border=' ', width=0, ps=0):
    slen = len(content)
    blen = 4 * len(border)
    if table_max_width - blen <= 0:
        print("too short window")
    if width == 0:
        width = table_max_width - blen
    width = min(width, table_max_width - blen)
    if slen == 1:
        print(content * (width + blen))
        return
    offset = 0
    conts = []
    clens = [width] * int(slen / width)
    if slen % width > 0:
        clens.append(slen % width)
    for clen in clens:
        toff = offset + clen
        conts.append(content[offset:toff])
        offset = toff
    for line in conts:  
        slen = len(line)
        if ps < 0:      
            splen = width - slen - 1
            if splen >= 0:
                print(border, line, sp * splen, border)
            else:
                print(border, line, border)
        elif ps == 0:
            llen = int((width - slen - 2) / 2) - 1
            if llen == 0:
                rlen = width - slen - 1
                if rlen > 0:
                    print(border, line, sp * rlen, border)
                else:
                    print(border, line, border)
            else:
                rlen = width - llen - slen - 2
                if rlen > 0:
                    print(border, sp * llen, line, sp * rlen, border)
                else:
                    print(border, sp * llen, line, border)
        else:
            splen = width - slen - 1
            if splen > 0:
                print(border, sp * splen, line, border)
            else:
                print(border, line, border)