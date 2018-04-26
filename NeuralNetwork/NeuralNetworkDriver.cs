using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class NeuralNetwork : Form
    {
        /// <summary>
        /// For overriding movement of window
        /// </summary>
        private const int WM_NCHITTEST = 0x84;
        private const int HT_CLIENT = 0x1;
        private const int HT_CAPTION = 0x2;
        /// <summary>
        /// Custom GUI offset
        /// </summary>
        private const int CLOSE_FORM_HORZ_OFFSET = 30;
        private const int PANEL_VERT_OFFSET = 25;
        private const int PICTUREBOX_OFFSET = 25;
        private const int PICTUREBOX_OFFSET_HALF = 12;
        private const int MAX_FORM_HORZ_OFFSET = 60;
        private const int MIN_FORM_HORZ_OFFSET = 90;
        private const int LEFT_OFFSET = 14;
        private const int LABEL_SIZE = 24;
        private const int BUTTON_GAP = 3;
        private const float BUTTON_FONT_SIZE = 8f;
        private const int AI_TEXTBOX_SIZE = 100;
        /// <summary>
        /// Main GUI variable
        /// </summary>
        private PictureBox pictureBoxOne;
        private PictureBox pictureBoxAI;
        private Panel aiPanel;
        private Panel panel;
        private CustomButton closeForm;
        private CustomButton minForm;
        private CustomButton maxForm;
        private Color themeColor;
        private Color themeBackgroundColor;
        private Color themeBackgroundColorTwo;
        private TextBox title;
        private CustomTextBox aiTextBox;
        private CustomButton aiDetermineButton;
        /// <summary>
        /// Variables for drawing
        /// </summary>
        private Graphics G;
        private Bitmap B;
        private SolidBrush brush;
        private const int LETTER_SIDE = 28;
        /// <summary>
        /// Bitmaps
        /// </summary>
        private Bitmap[] data;
        private byte[] labelsDigit;
        private char[] labelsLetter;
        /// <summary>
        /// Neural networks
        /// </summary>
        private NeuralNetwork<byte> digitNN;
        private NeuralNetwork<char> letterNN;
        public static bool digit;
        public static bool letter;
        /// <summary>
        /// Constants
        /// </summary>
        private const int INT_SIZE = 4;
        private const int DOUBLE_SIZE = 8;

        public NeuralNetwork()
        {
            this.themeBackgroundColor = Color.FromArgb(175, 0, 0, 0);
            this.themeBackgroundColorTwo = Color.FromArgb(100, 0, 0, 0);
            this.themeColor = Color.FromArgb(200, 144, 238, 144);
            this.brush = new SolidBrush(Color.Cyan);
            InitializeComponent();
            InitializeCustom();
            InitializeDrawing();
            CustomizeMenuStrip(menuStrip1);
            digit = false;
            // testing
            //test();
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None;
        }
        
        public void AddTextToAITextBoxMethod(string text)
        {

        }

        /// <summary>
        /// Initialize drawings for panel
        /// </summary>
        private void InitializeDrawing()
        {
            B = new Bitmap(28, 28);
            G = Graphics.FromImage(B);
        }

        /// <summary>
        /// Handles drawing on mouse move while left click
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void pictureBoxOne_MouseMove(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                int cX = (int)((double)e.X / (double)pictureBoxOne.Width * LETTER_SIDE);
                int cY = (int)((double)e.Y / (double)pictureBoxOne.Width * LETTER_SIDE);
                G.DrawEllipse(Pens.Cyan, new Rectangle(new Point(cX, cY), new Size(2, 2)));
            }
            pictureBoxOne.Image = B;
        }

        /// <summary>
        /// Initializes custom GUI
        /// </summary>
        private void InitializeCustom()
        {
            //
            // Control
            //
            Width = 1000;
            Height = 600;
            // 
            // pictureBoxOne shown on the left
            //
            pictureBoxOne = new PictureBox();
            pictureBoxOne.Location = new System.Drawing.Point(PICTUREBOX_OFFSET, 27 + PICTUREBOX_OFFSET);
            pictureBoxOne.Name = "pictureBoxOne";
            pictureBoxOne.TabStop = false;
            pictureBoxOne.BackColor = themeBackgroundColorTwo;
            pictureBoxOne.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxOne.MouseMove += new MouseEventHandler(this.pictureBoxOne_MouseMove);
            // 
            // ai panel shown on the right
            //
            aiPanel = new Panel();
            aiPanel.Name = "aiPanel";
            aiPanel.TabStop = false;
            aiPanel.BackColor = themeBackgroundColorTwo;
            //
            // pictureBoxAI in ai panel
            //
            pictureBoxAI = new PictureBox();
            pictureBoxAI.Location = new Point(PICTUREBOX_OFFSET_HALF, PICTUREBOX_OFFSET_HALF);
            pictureBoxAI.Name = "pictureBoxAI";
            pictureBoxAI.BackColor = Color.Black;
            pictureBoxAI.SizeMode = PictureBoxSizeMode.Zoom;
            aiPanel.Controls.Add(pictureBoxAI);
            //
            // aiTextBox
            //
            aiTextBox = new CustomTextBox();
            aiTextBox.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(35)))), ((int)(((byte)(35)))), ((int)(((byte)(35)))));
            aiTextBox.BorderStyle = System.Windows.Forms.BorderStyle.None;
            aiTextBox.Font = new System.Drawing.Font("Microsoft YaHei UI", 9F,
                System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            aiTextBox.Name = "Title";
            aiTextBox.ReadOnly = true;
            aiTextBox.Enabled = false;
            aiTextBox.TabStop = false;
            aiTextBox.Multiline = true;
            aiTextBox.Text = "Hello there! :D";
            aiPanel.Controls.Add(aiTextBox);
            //
            // aiDetermineButton
            //
            aiDetermineButton = new CustomButton();
            aiDetermineButton.ForeColor = Color.Cyan;
            aiDetermineButton.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            aiDetermineButton.BackColor = aiTextBox.BackColor;
            aiDetermineButton.FlatAppearance.BorderSize = 0;
            aiDetermineButton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            aiDetermineButton.Name = "aiDetermineButton";
            aiDetermineButton.TabStop = false;
            aiDetermineButton.Text = "\u2753";
            aiDetermineButton.UseMnemonic = false;
            aiDetermineButton.UseVisualStyleBackColor = true;
            aiDetermineButton.Click += new System.EventHandler(aiDetermineButton_Click);
            aiPanel.Controls.Add(aiDetermineButton);
            //
            // conext menu for pictureBoxOne
            //
            ContextMenu cm = new ContextMenu();
            MenuItem mnuClearImage = new MenuItem("Clear Drawing");
            mnuClearImage.Click += new EventHandler(mnuClearImage_Click);
            cm.MenuItems.Add(mnuClearImage);
            pictureBoxOne.ContextMenu = cm;
            // 
            // panel1
            // 
            panel = new Panel();
            panel.BackColor = System.Drawing.Color.Transparent;
            panel.Controls.Add(this.menuStrip1);
            panel.Location = new System.Drawing.Point(0, 25);
            panel.Name = "panel1";
            panel.Size = new System.Drawing.Size(this.Width, this.Height - 25);
            panel.Controls.Add(pictureBoxOne);
            panel.Controls.Add(aiPanel);
            // 
            // closeForm
            // 
            closeForm = new CustomButton();
            closeForm.ForeColor = themeColor;
            closeForm.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            closeForm.FlatAppearance.BorderSize = 0;
            closeForm.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            closeForm.Location = new System.Drawing.Point(this.Width - 45, 0);
            closeForm.Name = "closeForm";
            closeForm.Size = new System.Drawing.Size(30, 25);
            closeForm.TabIndex = 6;
            closeForm.Text = "X";
            closeForm.UseVisualStyleBackColor = true;
            closeForm.Click += new System.EventHandler(closeForm_Click);
            // 
            // maxForm
            // 
            maxForm = new CustomButton();
            maxForm.ForeColor = themeColor;
            maxForm.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            maxForm.FlatAppearance.BorderSize = 0;
            maxForm.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            maxForm.Location = new System.Drawing.Point(this.Width - 75, 0);
            maxForm.Name = "maxForm";
            maxForm.Size = new System.Drawing.Size(30, 25);
            maxForm.TabIndex = 5;
            maxForm.TabStop = false;
            maxForm.Text = "⎕";
            maxForm.UseMnemonic = false;
            maxForm.UseVisualStyleBackColor = true;
            maxForm.Click += new System.EventHandler(maxForm_Click);
            // 
            // minForm
            // 
            minForm = new CustomButton();
            minForm.ForeColor = themeColor;
            minForm.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            minForm.FlatAppearance.BorderSize = 0;
            minForm.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            minForm.Font = new System.Drawing.Font("Microsoft Sans Serif",
                BUTTON_FONT_SIZE, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            minForm.Location = new System.Drawing.Point(this.Width - 105, 0);
            minForm.Name = "minForm";
            minForm.Size = new System.Drawing.Size(30, 25);
            minForm.TabIndex = 4;
            minForm.TabStop = false;
            minForm.Text = "_";
            minForm.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            minForm.UseMnemonic = false;
            minForm.UseVisualStyleBackColor = true;
            minForm.Click += new System.EventHandler(minForm_Click);
            //
            // title
            //
            title = new TextBox();
            title.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(35)))), ((int)(((byte)(35)))), ((int)(((byte)(35)))));
            title.BorderStyle = System.Windows.Forms.BorderStyle.None;
            title.Font = new System.Drawing.Font("Microsoft YaHei UI", 9F,
                System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            title.Location = new System.Drawing.Point(LEFT_OFFSET, 4);
            title.Name = "Title";
            title.Size = new System.Drawing.Size(163, 16);
            title.ReadOnly = true;
            title.Enabled = false;
            title.TabStop = false;
            title.ForeColor = themeColor;
            title.Text = "Neural Network";
            //
            // ImageCompressor
            //
            Resize += new System.EventHandler(this.WindowResize);
            Controls.Add(panel);
            Controls.Add(maxForm);
            Controls.Add(title);
            Controls.Add(minForm);
            Controls.Add(closeForm);
            BackColor = Color.FromArgb(35, 35, 35);
        }

        /// <summary>
        /// Redraw form when resized
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void WindowResize(object sender, EventArgs e)
        {
            if (this.WindowState != FormWindowState.Minimized)
            {
                Control control = (Control)sender;
                int w = control.Size.Width;
                int h = control.Size.Height;
                closeForm.Location = new Point(w - CLOSE_FORM_HORZ_OFFSET, 0);
                panel.Size = new Size(w, h - PANEL_VERT_OFFSET);
                maxForm.Location = new Point(w - MAX_FORM_HORZ_OFFSET, 0);
                minForm.Location = new Point(w - MIN_FORM_HORZ_OFFSET, 0);
                pictureBoxOne.Size = new Size(w / 2 - PICTUREBOX_OFFSET * 3 / 2, h - 54 - PICTUREBOX_OFFSET * 2);
                aiPanel.Size = new Size(w / 2 - PICTUREBOX_OFFSET * 3 / 2, h - 54 - PICTUREBOX_OFFSET * 2);
                aiPanel.Location = new Point(w / 2 + PICTUREBOX_OFFSET_HALF, 27 + PICTUREBOX_OFFSET);
                pictureBoxAI.Size = new Size(aiPanel.Size.Width - PICTUREBOX_OFFSET_HALF * 2, aiPanel.Size.Height - PICTUREBOX_OFFSET_HALF * 3 - AI_TEXTBOX_SIZE);
                aiTextBox.Location = new Point(PICTUREBOX_OFFSET_HALF, pictureBoxAI.Size.Height + PICTUREBOX_OFFSET_HALF * 2);
                aiTextBox.Size = new Size((int)(pictureBoxAI.Size.Width * 0.9f) - PICTUREBOX_OFFSET_HALF, AI_TEXTBOX_SIZE);
                aiDetermineButton.Location = new Point(aiTextBox.Size.Width + PICTUREBOX_OFFSET_HALF * 2, aiTextBox.Location.Y);
                aiDetermineButton.Size = new Size(pictureBoxAI.Size.Width - aiTextBox.Size.Width - PICTUREBOX_OFFSET_HALF, aiTextBox.Size.Height);
            }
        }

        /// <summary>
        /// Clear the images on pictureBoxOne
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void mnuClearImage_Click(object sender, EventArgs e)
        {
            G.Clear(Color.Transparent);
        }

        /// <summary>
        /// Closes the form when closeForm button is clicked
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void closeForm_Click(object sender, EventArgs e)
        {
            Environment.Exit(0);
        }

        /// <summary>
        /// Minimizes the form when minForm button is clicked
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void minForm_Click(object sender, EventArgs e)
        {
            this.WindowState = FormWindowState.Minimized;
        }

        /// <summary>
        /// Maximize the form when maxForm button is clicked
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void maxForm_Click(object sender, EventArgs e)
        {
            if (this.WindowState == FormWindowState.Maximized)
            {
                this.WindowState = FormWindowState.Normal;
            }
            else
            {
                this.WindowState = FormWindowState.Maximized;
            }
        }

        protected override void WndProc(ref Message m)
        {
            base.WndProc(ref m);
            if (m.Msg == WM_NCHITTEST)
            {
                m.Result = (IntPtr)(HT_CAPTION);
            }
        }

        /// <summary>
        /// Custom tool strip renderer
        /// </summary>
        private class MyRenderer : ToolStripProfessionalRenderer
        {
            public MyRenderer(Color themeBackgroundColor) : base(new MyColors(themeBackgroundColor)) { }

            protected override void OnRenderArrow(ToolStripArrowRenderEventArgs e)
            {
                var toolStripMenuItem = e.Item as ToolStripMenuItem;
                if (toolStripMenuItem != null)
                {
                    e.ArrowColor = Color.FromArgb(200, 144, 238, 144);
                }
                base.OnRenderArrow(e);
            }
        }

        /// <summary>
        /// Class that override specific form item color
        /// </summary>
        private class MyColors : ProfessionalColorTable
        {
            private Color themeBackgroundColor;

            public MyColors(Color themeBackgroundColor)
            {
                this.themeBackgroundColor = themeBackgroundColor;
            }

            public override Color MenuItemSelected
            {
                get { return themeBackgroundColor; }
            }
            public override Color ButtonSelectedGradientMiddle
            {
                get { return Color.Transparent; }
            }

            public override Color ButtonSelectedHighlight
            {
                get { return Color.Transparent; }
            }

            public override Color ButtonCheckedGradientBegin
            {
                get { return themeBackgroundColor; }
            }
            public override Color ButtonCheckedGradientEnd
            {
                get { return themeBackgroundColor; }
            }
            public override Color ButtonSelectedBorder
            {
                get { return Color.FromArgb(200, 144, 238, 144); }
            }
            public override Color ToolStripDropDownBackground
            {
                get { return themeBackgroundColor; }
            }
            public override Color CheckSelectedBackground
            {
                get { return themeBackgroundColor; }
            }
            public override Color MenuItemSelectedGradientBegin
            {
                get { return themeBackgroundColor; }
            }
            public override Color MenuItemSelectedGradientEnd
            {
                get { return themeBackgroundColor; }
            }
            public override Color MenuItemBorder
            {
                get { return Color.Black; }
            }
            public override Color MenuItemPressedGradientBegin
            {
                get { return Color.Transparent; }
            }
            public override Color CheckBackground
            {
                get { return themeBackgroundColor; }
            }
            public override Color CheckPressedBackground
            {
                get { return themeBackgroundColor; }
            }
            public override Color ImageMarginGradientBegin
            {
                get { return Color.Transparent; }
            }
            public override Color ImageMarginGradientMiddle
            {
                get { return Color.Transparent; }
            }
            public override Color ImageMarginGradientEnd
            {
                get { return Color.Transparent; }
            }
            public override Color MenuItemPressedGradientEnd
            {
                get { return Color.Transparent; }
            }
        }

        /// <summary>
        /// Custom button that override Button
        /// </summary>
        public class CustomButton : Button
        {
            protected override bool ShowFocusCues
            {
                get
                {
                    return false;
                }
            }
        }

        public class CustomTextBox : TextBox
        {
            public CustomTextBox()
            {
                this.SetStyle(ControlStyles.UserPaint, true);

                //InitializeComponent();
            }
            protected override void OnPaint(PaintEventArgs e)
            {
                SolidBrush drawBrush = new SolidBrush(Color.Cyan); //Use the ForeColor property
                                                                  // Draw string to screen.
                e.Graphics.DrawString(Text, Font, drawBrush, 0f, 0f); //Use the Font property
            }
        }

        /// <summary>
        /// Customize menuStrip's color
        /// </summary>
        /// <param name="menuStrip">Menustrip object</param>
        private void CustomizeMenuStrip(MenuStrip menuStrip)
        {
            menuStrip.Renderer = new MyRenderer(themeBackgroundColor);
            menuStrip.BackColor = Color.Transparent;
            menuStrip.ForeColor = themeColor;
            openToolStripMenuItem.ForeColor = themeColor;
            saveToolStripMenuItem.ForeColor = themeColor;
            aiToolStripMenuItem.ForeColor = themeColor;
            newToolStripMenuItem.ForeColor = themeColor;
            loadToolStripMenuItem.ForeColor = themeColor;
            trainToolStripMenuItem.ForeColor = themeColor;
            digitToolStripMenuItem.ForeColor = themeColor;
            saveToolStripMenuItem1.ForeColor = themeColor;
            letterToolStripMenuItem.ForeColor = themeColor;
            newLetterToolStripMenuItem.ForeColor = themeColor;
            loadLetterToolStripMenuItem.ForeColor = themeColor;
            trainLetterToolStripMenuItem1.ForeColor = themeColor;
            saveLetterToolStripMenuItem2.ForeColor = themeColor;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Open Data";
                string path = "";
                string secondPath = "";
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    path = dialog.FileName;
                }
                if (dialog.ShowDialog() == DialogResult.OK)
                {
                    secondPath = dialog.FileName;
                }
                if (path.Contains("digit-train-labels") &&
                    secondPath.Contains("digit-train-images"))
                {
                    pictureBoxAI.Image = Properties.Resources.loading;
                    Thread t = new Thread(() => MnistLoader.LoadDigitMnist(path, secondPath, out data, out labelsDigit, pictureBoxAI));
                    t.Start();
                }
                else if (secondPath.Contains("digit-train-labels") &&
                    path.Contains("digit-train-images"))
                {
                    pictureBoxAI.Image = Properties.Resources.loading;
                    Thread t = new Thread(() => MnistLoader.LoadDigitMnist(secondPath, path, out data, out labelsDigit, pictureBoxAI));
                    t.Start();
                }
                if (path.Contains("emnist-balanced-train-labels") &&
                    secondPath.Contains("emnist-balanced-train-images"))
                {
                    pictureBoxAI.Image = Properties.Resources.loading;
                    Thread t = new Thread(() => MnistLoader.LoadLetterMnist(path, secondPath, out data, out labelsLetter, pictureBoxAI));
                    t.Start();
                }
                else if (secondPath.Contains("emnist-balanced-train-labels") &&
                    path.Contains("emnist-balanced-train-images"))
                {
                    pictureBoxAI.Image = Properties.Resources.loading;
                    Thread t = new Thread(() => MnistLoader.LoadLetterMnist(secondPath, path, out data, out labelsLetter, pictureBoxAI));
                    t.Start();
                }
            }
        }

        /// <summary>
        /// Determine the drawn image and determine the output
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void aiDetermineButton_Click(object sender, EventArgs e)
        {
            if (digitNN != null)
            {
                double[] input = DigitNN.transformBitmapdata(B);
                byte result = digitNN.determine(input);
                if (aiTextBox.Lines.Length == 6)
                {
                    NeuralNetwork.CustomTextBox temp = new NeuralNetwork.CustomTextBox();
                    for (int i = 1; i < aiTextBox.Lines.Length; i++)
                    {
                        temp.Text += aiTextBox.Lines[i] + "\n";
                    }
                    temp.Text += "I think it is a " + result + " :)";
                    aiTextBox.Text = temp.Text;
                }
                else
                {
                    aiTextBox.Text += "\nI think it is a " + result + " :)";
                }
            }
            if (letterNN != null)
            {
                double[] input = LetterNN.transformBitmapdata(B);
                char result = letterNN.determine(input);
                if (aiTextBox.Lines.Length == 6)
                {
                    NeuralNetwork.CustomTextBox temp = new NeuralNetwork.CustomTextBox();
                    for (int i = 1; i < aiTextBox.Lines.Length; i++)
                    {
                        temp.Text += aiTextBox.Lines[i] + "\n";
                    }
                    temp.Text += "I think it is a " + result + " :)";
                    aiTextBox.Text = temp.Text;
                }
                else
                {
                    aiTextBox.Text += "\nI think it is a " + result + " :)";
                }
            }
        }

        /// <summary>
        /// Create a new digit neural network
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void newDigitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int[] layers = new int[3] { 784, 100, 10 };
            DigitNN.init(out digitNN, layers);
            pictureBoxAI.Image = Properties.Resources.ai;
            digit = true;
        }

        /// <summary>
        /// Train neural networks of digit type
        /// </summary>
        /// <param name="sender">sender object</param>
        /// <param name="e">event</param>
        private void trainDigitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (data != null && labelsDigit != null && digitNN != null)
            {
                Thread t = new Thread(trainDigitThread);
                pictureBoxAI.Image = Properties.Resources.loading;
                t.Start();
            }
            else
            {
                // handle if data not yet loaded or neural network not yet initalized
            }
        }

        /// <summary>
        /// Thread for letter training
        /// </summary>
        private void trainLetterThread()
        {

            pictureBoxOne.Image = data[0];
            aiTextBox.Invoke(new Action(() =>
            {
                aiTextBox.Text += "\n" + labelsLetter[0];
            }));

            double[][] transformedData = LetterNN.transformBitmapdata(data);
            int epochs = 10000;
            int batchSize = 47;
            double[][] trainData = null;
            double[][] validset = null;
            char[] trainLabels = null;
            char[] validsetLabels = null;
            int testDataLength = 70000;
            if (labelsLetter.Length == 88800)
            {
                LetterNN.splitIntoTrainAndValidset(transformedData, out trainData, out validset, testDataLength);
                LetterNN.splitIntoTrainAndValidsetLabels(labelsLetter, out trainLabels, out validsetLabels, labelsLetter.Length - testDataLength);
            }
            Train<char>.trainNetwork(letterNN, trainData, validset, labelsLetter, validsetLabels, epochs, batchSize, aiTextBox);
            pictureBoxAI.Image = Properties.Resources.ai;
        }

        /// <summary>
        /// Thread for digit training
        /// </summary>
        private void trainDigitThread()
        {
            double[][] transformedData = DigitNN.transformBitmapdata(data);
            int epochs = 30;
            int batchSize = 10;
            double[][] trainData = null;
            double[][] validset = null;
            byte[] trainLabels = null;
            byte[] validsetLabels = null;
            if (labelsDigit.Length == 10000)
            {
                DigitNN.splitIntoTrainAndValidset(transformedData, out trainData, out validset, labelsDigit.Length - labelsDigit.Length / 10);
                DigitNN.splitIntoTrainAndValidsetLabels(labelsDigit, out trainLabels, out validsetLabels, labelsDigit.Length - labelsDigit.Length / 10);

            }
            if (labelsDigit.Length == 60000)
            {
                DigitNN.splitIntoTrainAndValidset(transformedData, out trainData, out validset, labelsDigit.Length - labelsDigit.Length / 6);
                DigitNN.splitIntoTrainAndValidsetLabels(labelsDigit, out trainLabels, out validsetLabels, labelsDigit.Length - labelsDigit.Length / 6);
            }
            Train<byte>.trainNetwork(digitNN, trainData, validset, labelsDigit, validsetLabels, epochs, batchSize, aiTextBox);
            pictureBoxAI.Image = Properties.Resources.ai;
        }

        private void savingWeights(NeuralNetwork<byte> nn)
        {
            SaveFileDialog save = new SaveFileDialog();
            save.FileName = "DefaultOutputName.txt";
            save.Filter = "Text File | *.txt";
            DialogResult dialogResult = saveFileDialog.ShowDialog();

            int layerLength = nn.Layers.Length;
            int wLength = 0;
            int bLength = 0;
            int arrayOffset = 0;
            byte[] savedArray;

            for (int l = 0; l < layerLength - 1; l++)
            {
                wLength += nn.Layers[l] * nn.Layers[l + 1];
            }

            for (int l = 1; l < layerLength; l++)
            {
                bLength += nn.Layers[l];
            }

            savedArray = new byte[wLength * DOUBLE_SIZE + bLength * DOUBLE_SIZE + layerLength * INT_SIZE + INT_SIZE];

            Buffer.BlockCopy(BitConverter.GetBytes(layerLength), 0, savedArray, arrayOffset, INT_SIZE);
            arrayOffset += INT_SIZE;

            for (int i = 0; i < layerLength; i++)
            {
                Buffer.BlockCopy(BitConverter.GetBytes(nn.Layers[i]), 0, savedArray, arrayOffset, INT_SIZE);
                arrayOffset += INT_SIZE;
            }

            if (dialogResult == DialogResult.OK)
            {
                Stream f = saveFileDialog.OpenFile();
                if (f != null)
                {
                    BinaryWriter wr = new BinaryWriter(f);

                    for (int i = 0; i < nn.Weights.Length; i++)
                    {
                        for (int j = 0; j < nn.Weights[i].Length; j++)
                        {
                            Buffer.BlockCopy(nn.Weights[i][j], 0, savedArray, arrayOffset, nn.Weights[i][j].Length * DOUBLE_SIZE);
                            arrayOffset += nn.Weights[i][j].Length * DOUBLE_SIZE;
                        }
                    }

                    for (int i = 0; i < nn.Biases.Length; i++)
                    {
                        Buffer.BlockCopy(nn.Biases[i], 0, savedArray, arrayOffset, nn.Biases[i].Length * DOUBLE_SIZE);
                        arrayOffset += nn.Biases[i].Length * DOUBLE_SIZE;
                    }
                    wr.Write(savedArray);
                    wr.Close();
                    f.Close();
                }
            }
        }

        /// <summary>
        /// Temporarily storing a char nn, to be merged with the byte one
        /// </summary>
        /// <param name="nn"></param>
        private void savingWeights(NeuralNetwork<char> nn)
        {
            SaveFileDialog save = new SaveFileDialog();
            save.FileName = "DefaultOutputName.txt";
            save.Filter = "Text File | *.txt";
            DialogResult dialogResult = saveFileDialog.ShowDialog();

            int layerLength = nn.Layers.Length;
            int wLength = 0;
            int bLength = 0;
            int arrayOffset = 0;
            byte[] savedArray;

            for (int l = 0; l < layerLength - 1; l++)
            {
                wLength += nn.Layers[l] * nn.Layers[l + 1];
            }

            for (int l = 1; l < layerLength; l++)
            {
                bLength += nn.Layers[l];
            }

            savedArray = new byte[wLength * DOUBLE_SIZE + bLength * DOUBLE_SIZE + layerLength * INT_SIZE + INT_SIZE];

            Buffer.BlockCopy(BitConverter.GetBytes(layerLength), 0, savedArray, arrayOffset, INT_SIZE);
            arrayOffset += INT_SIZE;

            for (int i = 0; i < layerLength; i++)
            {
                Buffer.BlockCopy(BitConverter.GetBytes(nn.Layers[i]), 0, savedArray, arrayOffset, INT_SIZE);
                arrayOffset += INT_SIZE;
            }

            if (dialogResult == DialogResult.OK)
            {
                Stream f = saveFileDialog.OpenFile();
                if (f != null)
                {
                    BinaryWriter wr = new BinaryWriter(f);

                    for (int i = 0; i < nn.Weights.Length; i++)
                    {
                        for (int j = 0; j < nn.Weights[i].Length; j++)
                        {
                            Buffer.BlockCopy(nn.Weights[i][j], 0, savedArray, arrayOffset, nn.Weights[i][j].Length * DOUBLE_SIZE);
                            arrayOffset += nn.Weights[i][j].Length * DOUBLE_SIZE;
                        }
                    }

                    for (int i = 0; i < nn.Biases.Length; i++)
                    {
                        Buffer.BlockCopy(nn.Biases[i], 0, savedArray, arrayOffset, nn.Biases[i].Length * DOUBLE_SIZE);
                        arrayOffset += nn.Biases[i].Length * DOUBLE_SIZE;
                    }
                    wr.Write(savedArray);
                    wr.Close();
                    f.Close();
                }
            }
        }

        /// <summary>
        /// Load digit ai
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void loadToolStripMenuItem_Click(object sender, EventArgs e)
        {
            double[][][] loadedWeights;
            double[][] loadedBiases;
            int[] header;
            int layerLength = 0;
            int wbOffset = 0;

            string file = "NULL";
            DialogResult result = openFileDialog.ShowDialog();
            if (result == DialogResult.OK) // Test result.
            {
                file = openFileDialog.FileName;

                byte[] fileByte = File.ReadAllBytes(file);

                layerLength = fileByte[0];
                header = new int[layerLength];
                wbOffset += INT_SIZE;

                for (int i = 0; i < layerLength; i++)
                {
                    header[i] = BitConverter.ToInt32(fileByte, INT_SIZE * i + INT_SIZE);
                    wbOffset += INT_SIZE;
                }
                //header length: 4, 784     100     100    10
                loadedWeights = new double[layerLength - 1][][];
                for (int i = 0; i < layerLength - 1; i++)
                {
                    loadedWeights[i] = new double[header[i + 1]][];
                    for (int j = 0; j < header[i + 1]; j++)
                    {
                        loadedWeights[i][j] = new double[header[i]];
                        Buffer.BlockCopy(fileByte, wbOffset, loadedWeights[i][j], 0, loadedWeights[i][j].Length * DOUBLE_SIZE);
                        wbOffset += loadedWeights[i][j].Length * DOUBLE_SIZE;
                    }
                }

                loadedBiases = new double[layerLength - 1][];
                for (int i = 0; i < layerLength - 1; i++)
                {
                    loadedBiases[i] = new double[header[i + 1]];
                    Buffer.BlockCopy(fileByte, wbOffset, loadedBiases[i], 0, loadedBiases[i].Length * DOUBLE_SIZE);
                    wbOffset += loadedBiases[i].Length * DOUBLE_SIZE;
                }
                digitNN = new NeuralNetwork<byte>(header);
                DigitNN.init(out digitNN, header);
                digitNN.testWB(ref loadedWeights, ref loadedBiases);
                pictureBoxAI.Image = Properties.Resources.ai;
                digit = true;
            }
        }

        private void saveToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            if (digitNN != null)
            {
                savingWeights(digitNN);
            }
        }

        private void newLetterToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int[] layers = new int[3] { 784, 470, 47 };
            LetterNN.init(out letterNN, layers);
            pictureBoxAI.Image = Properties.Resources.ai;
            digit = true;
        }

        private void trainLetterToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            if (data != null && labelsLetter != null && letterNN != null)
            {
                Thread t = new Thread(trainLetterThread);
                pictureBoxAI.Image = Properties.Resources.loading;
                t.Start();
            }
            else
            {
                // handle if data not yet loaded or neural network not yet initalized
            }
        }

        private void saveLetterToolStripMenuItem2_Click(object sender, EventArgs e)
        {
            if (letterNN != null)
            {
                savingWeights(letterNN);
            }
        }

        private void loadLetterToolStripMenuItem_Click(object sender, EventArgs e)
        {

            double[][][] loadedWeights;
            double[][] loadedBiases;
            int[] header;
            int layerLength = 0;
            int wbOffset = 0;

            string file = "NULL";
            DialogResult result = openFileDialog.ShowDialog();
            if (result == DialogResult.OK) // Test result.
            {
                file = openFileDialog.FileName;

                byte[] fileByte = File.ReadAllBytes(file);

                layerLength = fileByte[0];
                header = new int[layerLength];
                wbOffset += INT_SIZE;

                for (int i = 0; i < layerLength; i++)
                {
                    header[i] = BitConverter.ToInt32(fileByte, INT_SIZE * i + INT_SIZE);
                    wbOffset += INT_SIZE;
                }
                //header length: 4, 784     100     100    10
                loadedWeights = new double[layerLength - 1][][];
                for (int i = 0; i < layerLength - 1; i++)
                {
                    loadedWeights[i] = new double[header[i + 1]][];
                    for (int j = 0; j < header[i + 1]; j++)
                    {
                        loadedWeights[i][j] = new double[header[i]];
                        Buffer.BlockCopy(fileByte, wbOffset, loadedWeights[i][j], 0, loadedWeights[i][j].Length * DOUBLE_SIZE);
                        wbOffset += loadedWeights[i][j].Length * DOUBLE_SIZE;
                    }
                }

                loadedBiases = new double[layerLength - 1][];
                for (int i = 0; i < layerLength - 1; i++)
                {
                    loadedBiases[i] = new double[header[i + 1]];
                    Buffer.BlockCopy(fileByte, wbOffset, loadedBiases[i], 0, loadedBiases[i].Length * DOUBLE_SIZE);
                    wbOffset += loadedBiases[i].Length * DOUBLE_SIZE;
                }
                letterNN = new NeuralNetwork<char>(header);
                LetterNN.init(out letterNN, header);
                letterNN.testWB(ref loadedWeights, ref loadedBiases);
                pictureBoxAI.Image = Properties.Resources.ai;
                digit = true;
            }
        }
    }

    public static class Extensions
    {
        public static void Invoke<TControlType>(this TControlType control, Action<TControlType> del)
            where TControlType : Control
        {
            if (control.InvokeRequired)
                control.Invoke(new Action(() => del(control)));
            else
                del(control);
        }
    }
}
