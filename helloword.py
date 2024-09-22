import os

class TextEditor:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                file.write('')

    def read_file(self):
        with open(self.filename, 'r') as file:
            return file.read()

    def write_file(self, content):
        with open(self.filename, 'w') as file:
            file.write(content)

    def append_to_file(self, content):
        with open(self.filename, 'a') as file:
            file.write(content)

    def insert_line(self, line_number, content):
        lines = self.read_file().splitlines()
        if line_number <= len(lines):
            lines.insert(line_number - 1, content)
        else:
            lines.append(content)
        self.write_file('\n'.join(lines))

    def delete_line(self, line_number):
        lines = self.read_file().splitlines()
        if line_number <= len(lines):
            del lines[line_number - 1]
        self.write_file('\n'.join(lines))

    def replace_line(self, line_number, content):
        lines = self.read_file().splitlines()
        if line_number <= len(lines):
            lines[line_number - 1] = content
        self.write_file('\n'.join(lines))

if __name__ == "__main__":
    editor = TextEditor("example.txt")
    editor.write_file("Hello, world!\nThis is a test.")
    print("Initial content:")
    print(editor.read_file())
    print("\nAppending to the file...")
    editor.append_to_file("\nThis is an additional line.")
    print("Updated content:")
    print(editor.read_file())
    print("\nInserting a new line at position 2...")
    editor.insert_line(2, "New inserted line")
    print("Updated content:")
    print(editor.read_file())
    print("\nDeleting line number 3...")
    editor.delete_line(3)
    print("Updated content:")
    print(editor.read_file())
    print("\nReplacing line number 1...")
    editor.replace_line(1, "Replaced line")
    print("Final content:")
    print(editor.read_file())
